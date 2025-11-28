"""
Distillation Trainer
Implements Knowledge Distillation from a Teacher (Wav2Vec2) to a Student (MobileNet/ResNet)
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.training.trainer import Trainer
from src.models.huggingface import Wav2VecWakeword

logger = logging.getLogger(__name__)

class DistillationTrainer(Trainer):
    """
    Trainer that adds Knowledge Distillation support.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.distillation_enabled = self.config.distillation.enabled
        if self.distillation_enabled:
            logger.info("Initializing Distillation Trainer")
            self._init_teacher()
        else:
            self.teacher = None

    def _init_teacher(self):
        """Initialize the teacher model"""
        dist_config = self.config.distillation
        
        # TODO: Support loading from checkpoint path
        # For now, we initialize a pretrained Wav2VecWakeword
        logger.info(f"Loading teacher model: {dist_config.teacher_architecture}")
        
        if dist_config.teacher_architecture == "wav2vec2":
            self.teacher = Wav2VecWakeword(
                num_classes=self.config.model.num_classes,
                pretrained=True,
                freeze_feature_extractor=True
            )
        else:
            # Fallback or other architectures
            logger.warning(f"Unknown teacher architecture: {dist_config.teacher_architecture}. Using Wav2Vec2.")
            self.teacher = Wav2VecWakeword(num_classes=self.config.model.num_classes)

        # Load weights if path provided
        if dist_config.teacher_model_path:
            logger.info(f"Loading teacher weights from {dist_config.teacher_model_path}")
            # Checkpoint loading logic here...
            checkpoint = torch.load(dist_config.teacher_model_path, map_location="cpu")
            if "model_state_dict" in checkpoint:
                self.teacher.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.teacher.load_state_dict(checkpoint)
        
        self.teacher.to(self.device)
        self.teacher.eval()
        
        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        logger.info("Teacher model initialized and frozen")

    def compute_loss(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor, 
        inputs: Optional[torch.Tensor] = None,
        processed_inputs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute loss with distillation component
        
        Loss = (1 - alpha) * StudentLoss + alpha * KL(Student || Teacher)
        """
        student_loss = super().compute_loss(outputs, targets, inputs, processed_inputs)
        
        if not self.distillation_enabled or self.teacher is None:
            return student_loss
            
        if inputs is None:
            # Should not happen if training_loop is correct
            logger.warning("Inputs not provided to compute_loss, skipping distillation")
            return student_loss
            
        # Teacher forward pass
        with torch.no_grad():
            # Teacher expects raw audio. 
            # Check if inputs are raw audio (1D/2D) or spectrograms (4D)
            if inputs.ndim == 4:
                 # If inputs are spectrograms, we can't feed them to Wav2Vec2.
                 # This happens if dataset returns features (NPY) or Trainer processed it but didn't keep raw.
                 # We rely on 'raw_inputs' passed from _run_epoch which should be raw audio 
                 # IF the dataset yielded raw audio.
                 # But if dataset yielded NPY features, 'raw_inputs' is also 4D.
                 # In that case, we cannot perform distillation with Wav2Vec2 (it needs raw audio).
                 
                 # Check if we can proceed
                 # For now, just return student loss to avoid crash
                 # But ideally we should warn once
                 return student_loss

            teacher_logits = self.teacher(inputs)

        # Distillation Loss (KL Divergence)
        T = self.config.distillation.temperature
        alpha = self.config.distillation.alpha
        
        # Soft targets
        soft_targets = F.log_softmax(teacher_logits / T, dim=1)
        soft_prob = F.log_softmax(outputs / T, dim=1)
        
        # KLDivLoss expects input in log-space. 
        # If log_target=True, target should also be in log-space.
        distillation_loss = F.kl_div(soft_prob, soft_targets, reduction="batchmean", log_target=True) * (T**2)
        
        # Combined loss
        total_loss = (1 - alpha) * student_loss + alpha * distillation_loss
        
        return total_loss