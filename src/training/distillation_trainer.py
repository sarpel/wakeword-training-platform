"""
Distillation Trainer.

Implements Knowledge Distillation from a Teacher (Wav2Vec2) to a Student (MobileNet/ResNet).
"""

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.architectures import create_model
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
        else:
            logger.info("Distillation INACTIVE (Standard Training)")
            self.teacher = None

    def _load_teacher_checkpoint(self, checkpoint_path: str) -> dict:
        """
        Safely load teacher checkpoint with validation.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Loaded checkpoint dictionary

        Raises:
            ValueError: If path is outside allowed directories
            FileNotFoundError: If checkpoint file doesn't exist
        """
        from pathlib import Path

        # Convert to absolute path
        checkpoint_path_obj = Path(checkpoint_path).resolve()

        # Define allowed directories (checkpoints, current project root)
        project_root = Path.cwd().resolve()
        allowed_dirs = [
            project_root / "checkpoints",
            project_root / "models",
            project_root,
        ]

        # Validate path is within allowed directories
        is_allowed = any(checkpoint_path_obj.is_relative_to(allowed_dir) for allowed_dir in allowed_dirs)

        if not is_allowed:
            raise ValueError(
                f"Teacher checkpoint must be in allowed directories:\n"
                f"  - {project_root / 'checkpoints'}\n"
                f"  - {project_root / 'models'}\n"
                f"  - {project_root}\n"
                f"Got: {checkpoint_path_obj}"
            )

        # Check file exists
        if not checkpoint_path_obj.exists():
            raise FileNotFoundError(
                f"Teacher checkpoint not found: {checkpoint_path_obj}\n" f"Please ensure the checkpoint file exists."
            )

        # Check file is actually a file (not directory)
        if not checkpoint_path_obj.is_file():
            raise ValueError(f"Teacher checkpoint path is not a file: {checkpoint_path_obj}")

        logger.info(f"Loading teacher checkpoint: {checkpoint_path_obj}")

        # Load checkpoint with security: weights_only=True (PyTorch 1.13+)
        # This prevents arbitrary code execution from malicious pickles
        try:
            checkpoint = torch.load(
                checkpoint_path_obj, map_location="cpu", weights_only=True  # SECURITY: Prevents code execution
            )
        except TypeError:
            # Fallback for older PyTorch versions that don't support weights_only
            # But we know requirements.txt says 2.1.2 so this should be fine.
            # Just in case user has older env:
            logger.warning("torch.load doesn't support weights_only=True, loading unsafely (upgrade PyTorch!)")
            checkpoint = torch.load(checkpoint_path_obj, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"Failed to load teacher checkpoint from {checkpoint_path_obj}: {e}") from e

        return checkpoint

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
                # Prepare inputs for this specific teacher
                # Wav2Vec2 needs raw audio, Conformer/others might need raw or features
                # For now, we assume teachers take 'inputs' (raw audio if dim=2)
                # But if teacher is standard CNN, it might need features.
                # Heuristic: if input dim is 2 and teacher is not wav2vec2, it might need processing.

                inputs_teacher = inputs.to(teacher_device)

                # Check if we need to extract features for non-wav2vec teachers
                if inputs_teacher.dim() == 2 and not isinstance(teacher, Wav2VecWakeword):
                    # Use student's audio processor but on teacher's device
                    # Actually, we should probably have a processor for the teacher if it differs
                    # For now, let's assume all teachers except wav2vec2 take features
                    # Or just run them through the student's processor
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
                    # Align student features with teacher's intermediate features
                    # This is complex as dimensions must match.
                    # We'll use student.embed() and teacher.embed() for alignment.
                    with torch.cuda.amp.autocast(
                        enabled=dist_config.teacher_mixed_precision and teacher_device.type == "cuda"
                    ):
                        teacher_features = teacher.embed(inputs_teacher) if hasattr(teacher, "embed") else None

                    if teacher_features is not None:
                        student_features = self.model.embed(processed_inputs)
                        # Project student features if dimensions differ
                        # For simplicity, we use MSE on pooled features
                        teacher_features = teacher_features.to(self.device)
                        if student_features.shape == teacher_features.shape:
                            feature_alignment_loss += F.mse_loss(student_features, teacher_features)
                        else:
                            # Use adaptive pooling to match if only spatial dims differ
                            # But if channel dim differs, we skip for now (would need a projector)
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
