"""
Integration Test for QAT Training Flow.
Verifies the transition from standard training to QAT and the final reporting.
"""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config.defaults import WakewordConfig
from src.models.architectures import create_model
from src.training.checkpoint_manager import CheckpointManager
from src.training.trainer import Trainer


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3)
        self.fc = nn.Linear(4 * 2 * 2, 2)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dequant(x)
        return x


@pytest.mark.integration
def test_qat_training_flow(tmp_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup config
    config = WakewordConfig()
    config.training.epochs = 2
    config.qat.enabled = True
    config.qat.start_epoch = 1  # Start QAT at second epoch

    # Setup model and data
    model = SimpleModel().to(device)

    dummy_data = torch.randn(10, 1, 4, 4)
    dummy_labels = torch.randint(0, 2, (10,))
    dataset = TensorDataset(dummy_data, dummy_labels)
    loader = DataLoader(dataset, batch_size=2)

    checkpoint_manager = CheckpointManager(checkpoint_dir=tmp_path)

    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=config,
        checkpoint_manager=checkpoint_manager,
        device=device,
    )

    # Run training
    results = trainer.train()

    assert results is not None
    assert "qat_report" in results

    # Verify that model has fake quants after training
    assert hasattr(trainer.model.conv, "weight_fake_quant")

    # Verify report contents
    report = results["qat_report"]
    assert "fp32_acc" in report
    assert "quant_acc" in report
    assert "drop" in report


if __name__ == "__main__":
    pytest.main([__file__])
