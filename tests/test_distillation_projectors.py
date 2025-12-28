"""
Tests for Learnable Projectors in Distillation
"""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from src.config.defaults import WakewordConfig
from src.training.distillation_trainer import DistillationTrainer


class MockTeacher(nn.Module):
    def __init__(self, feat_dim=768):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        self.feat_dim = feat_dim

    def forward(self, x):
        # Handle both raw audio (B, T) and features (B, C, H, W)
        batch_size = x.size(0)
        return self.fc(torch.randn(batch_size, 10))

    def embed(self, x, layer_index=None):
        return torch.randn(x.size(0), self.feat_dim)


class MockWav2Vec(MockTeacher):
    pass


class MockStudent(nn.Module):
    def __init__(self, feat_dim=64):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        self.feat_dim = feat_dim

    def forward(self, x):
        return self.fc(torch.randn(x.size(0), 10))

    def embed(self, x, layer_index=None):
        return torch.randn(x.size(0), self.feat_dim)


class TestDistillationProjectors:
    """Test suite for feature alignment with learnable projectors"""

    @pytest.mark.unit
    def test_projector_initialization(self):
        """Test that projectors are initialized when dimensions mismatch"""
        config = WakewordConfig()
        config.distillation.enabled = True
        config.distillation.feature_alignment_enabled = True
        config.distillation.teacher_architecture = "conformer"  # Use one teacher for simplicity
        config.distillation.alignment_layers = [1]

        # Student dim 64, Teacher dim 768
        student = MockStudent(feat_dim=64)
        teacher = MockTeacher(feat_dim=768)

        # Mock dependencies to return our mock teacher
        import src.training.distillation_trainer

        original_create = src.training.distillation_trainer.create_model
        src.training.distillation_trainer.create_model = lambda *args, **kwargs: teacher

        try:
            trainer = DistillationTrainer(
                model=student,
                train_loader=MagicMock(),
                val_loader=MagicMock(),
                config=config,
                checkpoint_manager=MagicMock(),
                device="cpu",
            )

            # Check if projectors were created
            assert hasattr(trainer, "projectors")
            assert len(trainer.projectors) > 0
            # Projector should map student (64) to teacher (768)
            proj = trainer.projectors["teacher1_1"]
            assert isinstance(proj, nn.Module)

        finally:
            src.training.distillation_trainer.create_model = original_create

    @pytest.mark.unit
    def test_feature_alignment_loss_with_projector(self):
        """Test that feature alignment loss is non-zero when projectors are used"""
        config = WakewordConfig()
        config.distillation.enabled = True
        config.distillation.feature_alignment_enabled = True
        config.distillation.feature_alignment_weight = 1.0
        config.distillation.teacher_architecture = "conformer"
        config.distillation.alignment_layers = [1]

        student = MockStudent(feat_dim=64)
        teacher = MockTeacher(feat_dim=768)

        import src.training.distillation_trainer

        original_create = src.training.distillation_trainer.create_model
        src.training.distillation_trainer.create_model = lambda *args, **kwargs: teacher

        try:
            trainer = DistillationTrainer(
                model=student,
                train_loader=MagicMock(),
                val_loader=MagicMock(),
                config=config,
                checkpoint_manager=MagicMock(),
                device="cpu",
            )

            # Force loss computation
            outputs = torch.randn(2, 2)
            targets = torch.tensor([0, 1])
            inputs = torch.randn(2, 16000)
            processed_inputs = torch.randn(2, 1, 64, 50)

            # We need to mock audio_processor since it's used in compute_loss
            trainer.audio_processor = MagicMock()
            # Ensure criterion returns a real tensor
            trainer.criterion = lambda o, t: torch.tensor(1.0, requires_grad=True)

            loss = trainer.compute_loss(outputs, targets, inputs=inputs, processed_inputs=processed_inputs)
            assert loss.item() > 0

        finally:
            src.training.distillation_trainer.create_model = original_create
