"""
Distillation Trainer.

Implements Knowledge Distillation from a Teacher (Wav2Vec2) to a Student (MobileNet/ResNet).
"""

import logging
from typing import Any, Optional

import torch
import torch.nn.functional as F

from src.models.huggingface import Wav2VecWakeword
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


class DistillationTrainer(Trainer):
    """Trainer that adds Knowledge Distillation support."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the DistillationTrainer."""
        super().__init__(*args, **kwargs)
        self.teacher: Optional[torch.nn.Module] = None

        self.distillation_enabled = self.config.distillation.enabled
        if self.distillation_enabled:
            logger.info("Initializing Distillation Trainer")
            self._init_teacher()
        else:
            self.teacher = None

    def _init_teacher(self) -> None:
        """Initialize the teacher model."""
        dist_config = self.config.distillation

        # TODO: Support loading from checkpoint path
        # For now, we initialize a pretrained Wav2VecWakeword
        logger.info(f"Loading teacher model: {dist_config.teacher_architecture}")

        if dist_config.teacher_architecture == "wav2vec2":
            self.teacher = Wav2VecWakeword(
                num_classes=self.config.model.num_classes, pretrained=True, freeze_feature_extractor=True
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

        if self.teacher:
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
        processed_inputs: Optional[torch.Tensor] = None,
        is_hard_negative: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute loss with distillation component.

        Loss = (1 - alpha) * StudentLoss + alpha * KL(Student || Teacher)

        Args:
            outputs: Student model predictions (logits)
            targets: Ground truth labels
            inputs: Raw audio waveform (for teacher model)
            processed_inputs: Processed features (for embedding extraction)
            is_hard_negative: Tensor indicating hard negative samples

        Returns:
            Combined loss (student + distillation)
        """
        # Compute student loss with hard negative weighting
        student_loss = super().compute_loss(outputs, targets, inputs, processed_inputs, is_hard_negative)

        if not self.distillation_enabled or self.teacher is None:
            return student_loss

        if inputs is None:
            # Should not happen if training_loop is correct
            logger.warning("Inputs not provided to compute_loss, skipping distillation")
            return student_loss

        # Check if inputs are raw audio (2D: batch, time)
        # If inputs are spectrograms (3D or 4D), we cannot use the teacher (which expects raw audio)
        if inputs.dim() > 2:
            return student_loss

        # Teacher forward pass
        with torch.no_grad():
            # Teacher expects raw audio. 'inputs' should contain raw audio.
            teacher_outputs = self.teacher(inputs)

            # Get logits (handle different return types from teacher model)
            if isinstance(teacher_outputs, dict):
                teacher_logits = teacher_outputs.get("logits", teacher_outputs)
            elif isinstance(teacher_outputs, tuple):
                teacher_logits = teacher_outputs[0]
            else:
                teacher_logits = teacher_outputs

        # Distillation Loss (KL Divergence)
        T = self.config.distillation.temperature
        alpha = self.config.distillation.alpha

        # Soft targets from teacher
        soft_targets = F.log_softmax(teacher_logits / T, dim=1)
        # Soft probabilities from student
        soft_prob = F.log_softmax(outputs / T, dim=1)

        # KLDivLoss expects input in log-space.
        # If log_target=True, target should also be in log-space.
        distillation_loss = F.kl_div(soft_prob, soft_targets, reduction="batchmean", log_target=True) * (T**2)

        # Combined loss
        total_loss = (1 - alpha) * student_loss + alpha * distillation_loss

        return total_loss
