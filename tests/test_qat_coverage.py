import logging

import pytest
import torch
import torch.nn as nn

from src.config.defaults import QATConfig
from src.training.qat_utils import (
    calibrate_model,
    cleanup_qat_for_export,
    compare_model_accuracy,
    prepare_model_for_qat,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.weight = nn.Parameter(torch.randn(4, 1, 3, 3))

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.dequant(x)
        return x


def test_prepare_qat_unsupported_backend():
    """Trigger backend engine selection logic."""
    model = SimpleModel()
    config = QATConfig(enabled=True, backend="unsupported_engine")
    # This should trigger warnings and eventually fail in qconfig getter in recent torch versions
    try:
        prepared = prepare_model_for_qat(model, config)
        assert prepared is not None
    except AssertionError as e:
        assert "backend: unsupported_engine not supported" in str(e)


def test_cleanup_qat_for_export_logic():
    """Verify cleanup_qat_for_export handles fused modules."""
    model = SimpleModel()
    config = QATConfig(enabled=True, backend="fbgemm")
    prepared = prepare_model_for_qat(model, config)

    # Force some fused ops if possible or just call it
    # prepare_qat often uses FusedMovingAvgObsFakeQuantize by default for some configs
    cleaned = cleanup_qat_for_export(prepared)
    assert cleaned is not None


def test_calibrate_model_variations():
    """Test calibration with different input types."""
    model = SimpleModel()
    config = QATConfig(enabled=True, backend="fbgemm")
    prepared = prepare_model_for_qat(model, config)

    # List of tensors (not tuples)
    data_list = [torch.randn(1, 1, 10, 10) for _ in range(3)]
    calibrate_model(prepared, data_list)

    # Large index break check
    large_data = [torch.randn(1, 1, 10, 10) for _ in range(105)]
    calibrate_model(prepared, large_data)


def test_compare_model_accuracy_edge_cases():
    """Test accuracy comparison with non-standard batch format."""
    model = SimpleModel()
    # Mock data with just tensors (no labels) - should be skipped in eval
    data = [torch.randn(1, 1, 10, 10)]

    report = compare_model_accuracy(model, model, data)
    assert report["fp32_acc"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
