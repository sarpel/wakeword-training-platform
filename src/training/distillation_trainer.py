"""
Distillation Trainer.

Implements Knowledge Distillation from a Teacher (Wav2Vec2) to a Student (MobileNet/ResNet).
"""

from typing import Any, Optional

from src.config.logger import get_logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.architectures import create_model
from src.models.huggingface import Wav2VecWakeword
from src.training.trainer import Trainer

logger = get_logger(__name__)


class Projector(nn.Module):
    """Aligns student features with teacher features."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DistillationTrainer(Trainer):
    """Trainer that adds Knowledge Distillation support."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the DistillationTrainer."""
        super().__init__(*args, **kwargs)
        self.teacher: Optional[torch.nn.Module] = None
        self.projectors = nn.ModuleList()

        self.distillation_enabled = self.config.distillation.enabled
        if self.distillation_enabled:
            arch = self.config.distillation.teacher_architecture
            if arch == "dual":
                arch = f"wav2vec2 + {self.config.distillation.secondary_teacher_architecture}"

            logger.info(
                f"Distillation ACTIVE (Teacher: {arch}, Alpha: {self.config.distillation.alpha}, T: {self.config.distillation.temperature})"
            )

            logger.debug(f"  Temperature Scheduler: {self.config.distillation.temperature_scheduler}")
            if self.config.distillation.feature_alignment_enabled:
                logger.debug(
                    f"  Feature Alignment: ENABLED (Weight: {self.config.distillation.feature_alignment_weight})"
                )
            self._init_teacher()

            # Add projectors to optimizer if created
            if self.projectors and self.optimizer:
                logger.info(f"Adding {len(self.projectors)} projectors to optimizer")
                self.optimizer.add_param_group({"params": self.projectors.parameters()})
        else:
            logger.info("Distillation INACTIVE (Standard Training)")
            self.teacher = None

    def _init_teacher(self) -> None:
        """Initialize the teacher model(s)."""
        dist_config = self.config.distillation
        self.teachers = nn.ModuleList()
        self.teacher_devices = []

        logger.debug(f"Loading teacher architecture: {dist_config.teacher_architecture}")

        def load_one_teacher(arch, path):
            if arch == "wav2vec2":
                t = Wav2VecWakeword(
                    num_classes=self.config.model.num_classes, pretrained=True, freeze_feature_extractor=True
                )
            else:
                # Use standard factory for other architectures (e.g. conformer)
                t = create_model(arch, num_classes=self.config.model.num_classes)

            if path:
                logger.debug(f"Loading teacher weights for {arch} from: {path}")
                checkpoint = self._load_teacher_checkpoint(path)
                if "model_state_dict" in checkpoint:
                    t.load_state_dict(checkpoint["model_state_dict"])
                else:
                    t.load_state_dict(checkpoint)
                logger.debug(f"Successfully loaded {arch} weights")

            t.eval()
            for param in t.parameters():
                param.requires_grad = False

            # Device placement: Default to GPU if available and not explicitly forced to CPU
            # However, for this high-perf track, we prioritize GPU.
            use_gpu = not dist_config.teacher_on_cpu and torch.cuda.is_available()

            if use_gpu:
                t.to(self.device)
                if self.device == "cuda" or (isinstance(self.device, str) and "cuda" in self.device):
                    t.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
                dev = torch.device(self.device) if isinstance(self.device, str) else self.device
                logger.debug(f"Teacher {arch} deployed on GPU ({dev})")
            else:
                t.to("cpu")
                dev = torch.device("cpu")
                logger.debug(f"Teacher {arch} deployed on CPU")

            if dist_config.teacher_mixed_precision and dev.type == "cuda":
                t.half()
                logger.debug(f"Teacher {arch} using FP16 (Mixed Precision)")

            # Initialize Projector if alignment enabled
            if dist_config.feature_alignment_enabled:
                self._init_projector(t, dev)

            return t, dev

        if dist_config.teacher_architecture == "dual":
            # Primary teacher
            t1, d1 = load_one_teacher("wav2vec2", dist_config.teacher_model_path)
            self.teachers.append(t1)
            self.teacher_devices.append(d1)

            # Secondary teacher
            t2, d2 = load_one_teacher(
                dist_config.secondary_teacher_architecture, dist_config.secondary_teacher_model_path
            )
            self.teachers.append(t2)
            self.teacher_devices.append(d2)
        else:
            t, d = load_one_teacher(dist_config.teacher_architecture, dist_config.teacher_model_path)
            self.teachers.append(t)
            self.teacher_devices.append(d)

        # For backward compatibility
        self.teacher = self.teachers[0]
        self.teacher_device = self.teacher_devices[0]

        logger.debug(f"Initialized {len(self.teachers)} teacher(s)")

    def _init_projector(self, teacher: nn.Module, teacher_device: torch.device) -> None:
        """Initialize a projector for a teacher if dimensions mismatch."""
        # Get dimensions
        # We'll use a dummy input to find the embedding dimension
        student_dim = self._get_embed_dim(self.model, self.device)
        teacher_dim = self._get_embed_dim(teacher, teacher_device)

        logger.debug(f"Aligning Student ({student_dim}) -> Teacher ({teacher_dim})")

        if student_dim != teacher_dim:
            logger.info(f"Creating learnable projector: {student_dim} -> {teacher_dim}")
            projector = Projector(student_dim, teacher_dim).to(self.device)
            self.projectors.append(projector)
        else:
            # Identity-like projector (no-op)
            self.projectors.append(nn.Identity())

    def _get_embed_dim(self, model: nn.Module, device: torch.device) -> int:
        """Helper to find the output dimension of model.embed()"""
        model.eval()
        with torch.no_grad():
            # Create a dummy input based on model type
            # Standard CNN student expects features (B, 1, H, W)
            # Wav2Vec expects raw audio (B, T)
            # We'll try to infer from model type or just try both
            try:
                # Try feature-like input first
                dummy = torch.zeros(1, 1, 64, 50).to(device)
                out = model.embed(dummy)
            except Exception:
                try:
                    # Try audio-like input
                    dummy = torch.zeros(1, 16000).to(device)
                    out = model.embed(dummy)
                except Exception:
                    # Fallback to a common dimension if we can't find it
                    # But ideally we should find it.
                    logger.warning(f"Could not automatically determine embed dim for {model.__class__.__name__}")
                    return 128  # Default

            return out.shape[-1]

    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        inputs: Optional[torch.Tensor] = None,
        processed_inputs: Optional[torch.Tensor] = None,
        is_hard_negative: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute loss with multi-teacher distillation and feature alignment.
        """
        student_loss = super().compute_loss(outputs, targets, inputs, processed_inputs, is_hard_negative)

        if not self.distillation_enabled or not self.teachers:
            return student_loss

        if inputs is None:
            return student_loss

        dist_config = self.config.distillation

        # Temperature scheduling
        T = float(dist_config.temperature)
        if dist_config.temperature_scheduler == "linear_decay":
            # Simple linear decay from T to 1.0
            progress = self.state.epoch / self.config.training.epochs
            T = T - (T - 1.0) * progress
        T = max(1.0, T)

        # Distillation from all teachers
        all_teacher_logits = []
        feature_alignment_loss = torch.tensor(0.0, device=self.device)

        for i, (teacher, teacher_device) in enumerate(zip(self.teachers, self.teacher_devices)):
            with torch.no_grad():
                inputs_teacher = inputs.to(teacher_device)

                # Check if we need to extract features for non-wav2vec teachers
                if inputs_teacher.dim() == 2 and not isinstance(teacher, Wav2VecWakeword):
                    if hasattr(self, "audio_processor") and hasattr(self.audio_processor, "to"):
                        self.audio_processor.to(teacher_device)
                        inputs_teacher = self.audio_processor(inputs_teacher)

                with torch.cuda.amp.autocast(
                    enabled=dist_config.teacher_mixed_precision and teacher_device.type == "cuda"
                ):
                    teacher_out = teacher(inputs_teacher)

                if isinstance(teacher_out, dict):
                    logits = teacher_out.get("logits", teacher_out)
                elif isinstance(teacher_out, tuple):
                    logits = teacher_out[0]
                else:
                    logits = teacher_out

                all_teacher_logits.append(logits.to(self.device))

                # Feature alignment (optional)
                if dist_config.feature_alignment_enabled and processed_inputs is not None:
                    with torch.cuda.amp.autocast(
                        enabled=dist_config.teacher_mixed_precision and teacher_device.type == "cuda"
                    ):
                        teacher_features = teacher.embed(inputs_teacher) if hasattr(teacher, "embed") else None

                    if teacher_features is not None:
                        student_features = self.model.embed(processed_inputs)
                        teacher_features = teacher_features.to(self.device)

                        # Use projector if it exists for this teacher
                        if i < len(self.projectors):
                            proj = self.projectors[i]
                            student_features = proj(student_features)

                        if student_features.shape == teacher_features.shape:
                            feature_alignment_loss += F.mse_loss(student_features, teacher_features)
                        else:
                            # Try to match spatial dimensions if only those differ
                            if student_features.dim() == teacher_features.dim():
                                # Simple adaptive pooling if spatial dims differ but channels match
                                # (though Projector handles channel mismatch)
                                pass

        if not all_teacher_logits:
            return student_loss

        # Average logits from all teachers (Ensemble Distillation)
        mean_teacher_logits = torch.stack(all_teacher_logits).mean(dim=0)

        # Distillation Loss (KL)
        soft_targets = F.log_softmax(mean_teacher_logits / T, dim=1)
        soft_prob = F.log_softmax(outputs / T, dim=1)
        dist_loss = F.kl_div(soft_prob, soft_targets, reduction="batchmean", log_target=True) * (T**2)

        # Final Combined Loss
        alpha = dist_config.alpha
        total_loss = (1 - alpha) * student_loss + alpha * dist_loss

        if dist_config.feature_alignment_enabled:
            total_loss += dist_config.feature_alignment_weight * feature_alignment_loss

        return total_loss
