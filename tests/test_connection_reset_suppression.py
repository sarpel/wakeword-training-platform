"""
Tests for ConnectionResetError suppression
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config.defaults import WakewordConfig
from src.training.checkpoint_manager import CheckpointManager
from src.training.trainer import Trainer


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x.mean(dim=(1, 2, 3)) if x.ndim == 4 else x)


@pytest.mark.unit
def test_trainer_suppresses_connection_reset():
    """Verify Trainer.train suppresses ConnectionResetError on Windows"""
    if sys.platform != "win32":
        pytest.skip("Windows-only test")

    # Setup dummy objects
    model = MockModel()
    config = WakewordConfig()
    config.training.epochs = 2

    dataset = TensorDataset(torch.randn(10, 1, 64, 50), torch.randint(0, 2, (10,)))
    loader = DataLoader(dataset, batch_size=2)

    checkpoint_manager = MagicMock(spec=CheckpointManager)
    checkpoint_manager.checkpoint_dir = Path("cache/test_checkpoints")

    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=config,
        checkpoint_manager=checkpoint_manager,
        device="cpu",
    )

    # Mock train_epoch to raise ConnectionResetError
    with patch("src.training.trainer.train_epoch", side_effect=ConnectionResetError("WinError 10054")):
        # Should not raise exception
        results = trainer.train()
        assert results["final_epoch"] >= 0
