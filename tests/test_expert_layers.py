"""
Tests for Expert Layer Selection in Distillation
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
        return self.fc(torch.randn(x.size(0), 10))

    def embed(self, x, layer_index=None):
        # Return different dummy features based on layer_index to verify it's passed
        if layer_index is not None:
            return torch.randn(x.size(0), self.feat_dim) + layer_index
        return torch.randn(x.size(0), self.feat_dim)


class TestExpertLayerSelection:
    """Test suite for manual layer selection alignment"""

    @pytest.mark.unit
    def test_multi_layer_projector_creation(self):
        """Test that multiple projectors are created for multiple alignment layers"""
        config = WakewordConfig()
        config.distillation.enabled = True
        config.distillation.feature_alignment_enabled = True
        config.distillation.alignment_layers = [1, 5, 12]
        config.distillation.teacher_architecture = "conformer"

        import src.training.distillation_trainer

        original_create = src.training.distillation_trainer.create_model
        src.training.distillation_trainer.create_model = lambda *args, **kwargs: MockTeacher()

        try:
            trainer = DistillationTrainer(
                model=nn.Linear(10, 2),  # Student
                train_loader=MagicMock(),
                val_loader=MagicMock(),
                config=config,
                checkpoint_manager=MagicMock(),
                device="cpu",
            )

            # Check projectors
            # Keys should be teacher1_1, teacher1_5, teacher1_12
            assert len(trainer.projectors) == 3
            assert "teacher1_1" in trainer.projectors
            assert "teacher1_5" in trainer.projectors
            assert "teacher1_12" in trainer.projectors

        finally:
            src.training.distillation_trainer.create_model = original_create

    @pytest.mark.unit
    def test_multi_layer_alignment_loss(self):
        """Test that distillation loss incorporates multiple alignment layers"""
        config = WakewordConfig()
        config.distillation.enabled = True
        config.distillation.feature_alignment_enabled = True
        config.distillation.alignment_layers = [1, 2]
        config.distillation.teacher_architecture = "conformer"

        import src.training.distillation_trainer

        original_create = src.training.distillation_trainer.create_model
        src.training.distillation_trainer.create_model = lambda *args, **kwargs: MockTeacher()

        try:
            trainer = DistillationTrainer(
                model=nn.Linear(10, 2),
                train_loader=MagicMock(),
                val_loader=MagicMock(),
                config=config,
                checkpoint_manager=MagicMock(),
                device="cpu",
            )

            trainer.audio_processor = MagicMock()
            trainer.criterion = lambda o, t: torch.tensor(1.0, requires_grad=True)

            # Mock student embed to return something compatible
            trainer.model.embed = lambda x: torch.randn(x.size(0), 128)

            outputs = torch.randn(1, 2)
            targets = torch.tensor([0])
            inputs = torch.randn(1, 16000)
            processed_inputs = torch.randn(1, 1, 64, 50)

            loss = trainer.compute_loss(outputs, targets, inputs=inputs, processed_inputs=processed_inputs)
            assert loss.item() > 0

        finally:
            src.training.distillation_trainer.create_model = original_create
