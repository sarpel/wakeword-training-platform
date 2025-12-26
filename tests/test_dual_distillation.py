from unittest.mock import MagicMock

import pytest
import torch

from src.config.defaults import WakewordConfig
from src.training.distillation_trainer import DistillationTrainer


class MockTeacher(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.fc = torch.nn.Linear(10, 2)

    def forward(self, x):
        # Handle both raw audio (B, T) and features (B, C, H, W)
        batch_size = x.size(0)
        return self.fc(torch.randn(batch_size, 10))

    def embed(self, x):
        return torch.randn(x.size(0), 128)


class MockWav2Vec(MockTeacher):
    pass


class MockStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(torch.randn(x.size(0), 10))

    def embed(self, x):
        return torch.randn(x.size(0), 128)


def test_distillation_trainer_init_dual():
    config = WakewordConfig()
    config.distillation.enabled = True
    config.distillation.teacher_architecture = "dual"
    config.distillation.secondary_teacher_architecture = "conformer"

    # Mock loaders
    train_loader = MagicMock()
    train_loader.dataset.get_class_weights.return_value = torch.ones(2)
    val_loader = MagicMock()
    checkpoint_manager = MagicMock()

    # We need to mock Wav2VecWakeword and create_model
    import src.training.distillation_trainer

    src.training.distillation_trainer.Wav2VecWakeword = MockWav2Vec
    src.training.distillation_trainer.create_model = lambda *args, **kwargs: MockTeacher()

    trainer = DistillationTrainer(
        model=torch.nn.Linear(10, 2),
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_manager=checkpoint_manager,
        device="cpu",
    )

    assert len(trainer.teachers) == 2
    assert trainer.distillation_enabled is True


def test_dual_distillation_loss():
    config = WakewordConfig()
    config.distillation.enabled = True
    config.distillation.teacher_architecture = "dual"
    config.distillation.alpha = 0.5

    # Mock dependencies
    import src.training.distillation_trainer

    src.training.distillation_trainer.Wav2VecWakeword = MockWav2Vec
    src.training.distillation_trainer.create_model = lambda *args, **kwargs: MockTeacher()

    # Mock loaders
    train_loader = MagicMock()
    train_loader.dataset.get_class_weights.return_value = torch.ones(2)
    val_loader = MagicMock()
    checkpoint_manager = MagicMock()

    # We need to mock Wav2VecWakeword and create_model
    import src.training.distillation_trainer

    src.training.distillation_trainer.Wav2VecWakeword = MockWav2Vec
    src.training.distillation_trainer.create_model = lambda *args, **kwargs: MockTeacher()

    trainer = DistillationTrainer(
        model=MockStudent(),
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_manager=MagicMock(),
        device="cpu",
    )

    outputs = torch.randn(2, 2)
    targets = torch.tensor([0, 1])
    inputs = torch.randn(2, 16000)  # Raw audio

    loss = trainer.compute_loss(outputs, targets, inputs=inputs)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0
