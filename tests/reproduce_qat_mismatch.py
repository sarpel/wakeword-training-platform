"""
Reproduction script for QAT channel mismatch
"""
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config.defaults import WakewordConfig
from src.data.processor import AudioProcessor
from src.models.architectures import create_model
from src.training.qat_utils import compare_model_accuracy


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # tiny_conv-like structure: expects [B, 1, F, T]
        self.conv = nn.Conv2d(1, 16, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        # inputs are (B, C, F, T) -> (1, 1, 64, 151)
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def test_qat_mismatch_fix():
    """Verify that passing AudioProcessor fixes the mismatch"""
    model = SimpleModel()
    config = WakewordConfig()
    config.data.audio_duration = 1.5
    config.data.n_mels = 64

    # Processor to convert raw audio (B, T) -> (B, 1, F, T)
    processor = AudioProcessor(config, device="cpu")

    # Dataset with raw audio
    raw_audio = torch.randn(2, 24000)  # 2 samples, 1.5s at 16k
    targets = torch.randint(0, 2, (2,))
    dataset = TensorDataset(raw_audio, targets)
    loader = DataLoader(dataset, batch_size=2)

    # This should now work without error
    results = compare_model_accuracy(
        model,
        model,  # Use same model for both FP32 and Quant in this dummy test
        loader,
        device="cpu",
        audio_processor=processor,
    )

    assert "fp32_acc" in results
    assert results["fp32_acc"] >= 0


if __name__ == "__main__":
    test_qat_mismatch_fix()
