"""
Distillation Trainer.

Implements Knowledge Distillation from a Teacher (Wav2Vec2) to a Student (MobileNet/ResNet).
"""

from typing import Any, List, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.logger import get_logger
from src.models.architectures import create_model
from src.models.huggingface import Wav2VecWakeword, WhisperWakeword
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

    teacher: Optional[nn.Module]
    teachers: nn.ModuleList
    teacher_devices: List[torch.device]
    projectors: nn.ModuleDict
    distillation_enabled: bool
    teacher_device: torch.device

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the DistillationTrainer."""
        super().__init__(*args, **kwargs)
        self.teacher: Optional[torch.nn.Module] = None
        self.projectors = nn.ModuleDict()

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
        self.projectors = nn.ModuleDict()  # Changed to ModuleDict for multi-layer support

        logger.debug(f"Loading teacher architecture: {dist_config.teacher_architecture}")

        def load_one_teacher(arch, path, teacher_id):
            if arch == "wav2vec2":
                t = Wav2VecWakeword(
                    num_classes=self.config.model.num_classes, pretrained=True, freeze_feature_extractor=True
                )
            elif arch == "whisper":
                # Whisper teacher - uses encoder-only for feature extraction
                t = WhisperWakeword(num_classes=self.config.model.num_classes, pretrained=True, freeze_encoder=True)
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

            # Device placement
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

            # Initialize Projectors for all alignment layers
            if dist_config.feature_alignment_enabled:
                for layer_idx in dist_config.alignment_layers:
                    self._init_projector(t, dev, f"{teacher_id}_{layer_idx}", layer_idx)

            return t, dev

        if dist_config.teacher_architecture == "dual":
            # Primary teacher
            t1, d1 = load_one_teacher("wav2vec2", dist_config.teacher_model_path, "teacher1")
            self.teachers.append(t1)
            self.teacher_devices.append(d1)

            # Secondary teacher
            t2, d2 = load_one_teacher(
                dist_config.secondary_teacher_architecture, dist_config.secondary_teacher_model_path, "teacher2"
            )
            self.teachers.append(t2)
            self.teacher_devices.append(d2)
        else:
            t, d = load_one_teacher(dist_config.teacher_architecture, dist_config.teacher_model_path, "teacher1")
            self.teachers.append(t)
            self.teacher_devices.append(d)

        # For backward compatibility
        self.teacher = self.teachers[0]
        self.teacher_device = self.teacher_devices[0]

        logger.debug(f"Initialized {len(self.teachers)} teacher(s)")

    def _init_projector(self, teacher: nn.Module, teacher_device: torch.device, key: str, layer_idx: int) -> None:
        """Initialize a projector for a specific teacher layer if dimensions mismatch."""
        student_dim = self._get_embed_dim(self.model, self.device)
        # Pass layer_idx to get embedding dimension of that specific layer
        teacher_dim = self._get_embed_dim(teacher, teacher_device, layer_idx)

        logger.debug(f"Aligning Student ({student_dim}) -> Teacher Layer {layer_idx} ({teacher_dim})")

        if student_dim != teacher_dim:
            logger.info(f"Creating learnable projector [{key}]: {student_dim} -> {teacher_dim}")
            projector = Projector(student_dim, teacher_dim).to(self.device)
            self.projectors[key] = projector
        else:
            self.projectors[key] = nn.Identity()

    def _get_embed_dim(self, model: nn.Module, device: Any, layer_index: Optional[int] = None) -> int:
        """Helper to find the output dimension of model.embed()"""
        model.eval()
        with torch.no_grad():
            try:
                # Try feature-like input first
                dummy = torch.zeros(1, 1, 64, 50).to(device)
                if layer_index is not None:
                    out = cast(Any, model).embed(dummy, layer_index=layer_index)
                else:
                    out = cast(Any, model).embed(dummy)
            except Exception:
                try:
                    # Try audio-like input
                    dummy = torch.zeros(1, 16000).to(device)
                    if layer_index is not None:
                        out = cast(Any, model).embed(dummy, layer_index=layer_index)
                    else:
                        out = cast(Any, model).embed(dummy)
                except Exception:
                    logger.warning(f"Could not automatically determine embed dim for {model.__class__.__name__}")
                    return 128

            return out.shape[-1]

    def _calculate_dynamic_weights(self, all_teacher_logits: List[torch.Tensor]) -> torch.Tensor:
        """
        Calculate dynamic weights for each teacher based on their confidence (inverse entropy).

        Args:
            all_teacher_logits: List of logits from each teacher

        Returns:
            Normalized weights for each teacher (num_teachers, batch_size)
        """
        if len(all_teacher_logits) < 2:
            return torch.ones(len(all_teacher_logits), all_teacher_logits[0].size(0), device=self.device)

        confidences = []
        for logits in all_teacher_logits:
            probs = F.softmax(logits, dim=1)
            # Entropy = -sum(p * log(p))
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            # Confidence is inverse of entropy (stabilized)
            confidence = 1.0 / (entropy + 1e-5)
            confidences.append(confidence)

        # Normalize weights across teachers using softmax to maintain diversity
        # (Softmax ensures weights sum to 1 and preserves relative differences)
        conf_stack = torch.stack(confidences, dim=0)  # (num_teachers, batch_size)
        weights = F.softmax(conf_stack, dim=0)

        return weights

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
            teacher_id = f"teacher{i+1}"
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

                # Defensive NaN check - skip teacher if output is corrupted
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logger.warning(f"NaN/Inf detected in teacher {i+1} output, skipping this teacher")
                    continue

                all_teacher_logits.append(logits.to(self.device))

                # Feature alignment (optional)
                if dist_config.feature_alignment_enabled and processed_inputs is not None:
                    for layer_idx in dist_config.alignment_layers:
                        with torch.cuda.amp.autocast(
                            enabled=dist_config.teacher_mixed_precision and teacher_device.type == "cuda"
                        ):
                            teacher_features = (
                                cast(Any, teacher).embed(inputs_teacher, layer_index=layer_idx)
                                if hasattr(teacher, "embed")
                                else None
                            )

                        if teacher_features is not None:
                            student_features = cast(Any, self.model).embed(processed_inputs)
                            teacher_features = teacher_features.to(self.device)

                            # Use projector for this teacher and layer
                            proj_key = f"{teacher_id}_{layer_idx}"
                            if proj_key in self.projectors:
                                student_features = self.projectors[proj_key](student_features)

                            if student_features.shape == teacher_features.shape:
                                feature_alignment_loss += F.mse_loss(student_features, teacher_features)

        if not all_teacher_logits:
            return student_loss

        # Weighted Ensemble Distillation
        if len(all_teacher_logits) > 1:
            weights = self._calculate_dynamic_weights(all_teacher_logits)
            mean_teacher_logits = torch.zeros_like(all_teacher_logits[0])
            for i, logits in enumerate(all_teacher_logits):
                w = weights[i].unsqueeze(1)  # (batch_size, 1)
                mean_teacher_logits += w * logits
        else:
            mean_teacher_logits = all_teacher_logits[0]

        # Distillation Loss (KL)
        # Clamp logits to prevent numerical instability in softmax
        mean_teacher_logits = torch.clamp(mean_teacher_logits, min=-100.0, max=100.0)
        outputs_clamped = torch.clamp(outputs, min=-100.0, max=100.0)

        soft_targets = F.log_softmax(mean_teacher_logits / T, dim=1)
        soft_prob = F.log_softmax(outputs_clamped / T, dim=1)
        dist_loss = F.kl_div(soft_prob, soft_targets, reduction="batchmean", log_target=True) * (T**2)

        # Final Combined Loss
        alpha = dist_config.alpha
        total_loss = (1 - alpha) * student_loss + alpha * dist_loss

        if dist_config.feature_alignment_enabled:
            total_loss += dist_config.feature_alignment_weight * feature_alignment_loss

        return total_loss
