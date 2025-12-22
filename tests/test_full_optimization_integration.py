import torch
import pytest
from src.training.distillation_trainer import DistillationTrainer
from src.config.defaults import WakewordConfig
from src.models.architectures import create_model
from src.training.checkpoint_manager import CheckpointManager
from pathlib import Path
from unittest.mock import MagicMock

def test_full_integration_pipeline(tmp_path):
    # This test verifies that all new components work together in a single epoch
    config = WakewordConfig()
    
    # Enable all new features
    config.distillation.enabled = True
    config.distillation.teacher_architecture = "dual"
    config.distillation.feature_alignment_enabled = True
    config.distillation.temperature_scheduler = "linear_decay"
    
    config.training.epochs = 1
    config.training.batch_size = 2
    config.optimizer.scheduler = "onecycle"
    
    # Mock loaders
    # We need real datasets because AudioProcessor will be used
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.files = [{"path": "fake.wav", "category": "positive"}] * 4
        def __len__(self): return 4
        def __getitem__(self, idx):
            # Return raw audio (1, 16000)
            return torch.randn(1, 16000), 1, {"path": "fake.wav", "category": "positive"}
        def get_class_weights(self): return torch.ones(2)

    dataset = DummyDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    
    # Mock teachers
    import src.training.distillation_trainer
    class MockTeacher(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.fc = torch.nn.Linear(10, 2)
        def forward(self, x): 
            return torch.randn(x.size(0), 2)
        def embed(self, x):
            return torch.randn(x.size(0), 128)

    src.training.distillation_trainer.Wav2VecWakeword = MockTeacher
    src.training.distillation_trainer.create_model = lambda *args, **kwargs: MockTeacher()
    
    # Student model
    class MockStudent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(1, 8, 3),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten()
            )
            self.classifier = torch.nn.Linear(8, 2)
        def forward(self, x):
            return self.classifier(self.features(x))
        def embed(self, x):
            return self.features(x).repeat(1, 16) # Match 128

    student = MockStudent()
    
    checkpoint_manager = CheckpointManager(tmp_path / "ckpts")
    
    trainer = DistillationTrainer(
        model=student,
        train_loader=loader,
        val_loader=loader,
        config=config,
        checkpoint_manager=checkpoint_manager,
        device="cpu"
    )
    
    # Run one epoch
    results = trainer.train()
    assert results["final_epoch"] == 0
    assert "val_pauc" in results["history"]
    assert len(results["history"]["val_pauc"]) == 1
