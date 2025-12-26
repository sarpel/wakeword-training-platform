from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config.defaults import WakewordConfig
from src.training.checkpoint_manager import CheckpointManager
from src.training.trainer import Trainer


def test_channels_last_logic_applied():
    """Verify that model conversion to channels_last is attempted on GPU"""
    # Create a mock model that tracks calls
    mock_model = MagicMock(spec=nn.Module)
    mock_model.parameters.return_value = [torch.randn(1, 1, 3, 3)]

    # We want to verify that .to(memory_format=torch.channels_last) is called
    # .to() returns the model itself or a new one
    mock_model.to.return_value = mock_model

    config = WakewordConfig()
    checkpoint_manager = CheckpointManager(Path("cache/test_checkpoints"))

    # Dummy data
    dataset = TensorDataset(torch.randn(8, 1, 64, 50), torch.randint(0, 2, (8,)))
    loader = DataLoader(dataset, batch_size=4)

    # Force CUDA available for the test
    with patch("torch.cuda.is_available", return_value=True), patch("src.training.trainer.enforce_cuda"):
        trainer = Trainer(
            model=mock_model,
            train_loader=loader,
            val_loader=loader,
            config=config,
            checkpoint_manager=checkpoint_manager,
            device="cuda",
        )

        # Verify .to(memory_format=torch.channels_last) was called
        # The call in trainer.py is: self.model = self.model.to(memory_format=torch.channels_last)
        # Note: positional args might be present (device)
        mock_model.to.assert_any_call(memory_format=torch.channels_last)


def test_vram_estimation_logic():
    """Verify the VRAM estimation math in cuda_utils"""
    from src.config.cuda_utils import CUDAValidator

    validator = CUDAValidator()

    # Wav2vec2 teacher (1.2GB) + Resnet18 student (0.2GB) + context (0.5GB) + batch overhead
    # 1.2 + 0.2 + 0.5 = 1.9 + batch
    est = validator.estimate_vram_footprint_gb(teacher_arch="wav2vec2", student_arch="resnet18", batch_size=64)

    assert est >= 1.9
    assert est < 4.0  # Should be around 2.5-3.0 GB
