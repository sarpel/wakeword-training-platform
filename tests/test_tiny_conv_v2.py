"""
Tests for TinyConv V2 (Depthwise Separable Convolutions)
"""
import pytest
import torch

from src.models.architectures import create_model


class TestTinyConvV2:
    """Test suite for TinyConv V2 optimization"""

    @pytest.mark.unit
    def test_tiny_conv_v2_forward_shape(self):
        """Test TinyConv V2 produces correct output shape"""
        # Test standard (default)
        model_std = create_model("tiny_conv", num_classes=2, use_depthwise=False)
        inputs = torch.randn(2, 1, 64, 50)
        outputs_std = model_std(inputs)
        assert outputs_std.shape == (2, 2)

        # Test depthwise
        model_dw = create_model("tiny_conv", num_classes=2, use_depthwise=True)
        outputs_dw = model_dw(inputs)
        assert outputs_dw.shape == (2, 2)

    @pytest.mark.unit
    def test_tiny_conv_v2_parameter_reduction(self):
        """Test that depthwise separable convolutions significantly reduce parameters"""
        num_classes = 2
        tcn_num_channels = [16, 32, 64, 64]

        model_std = create_model(
            "tiny_conv", num_classes=num_classes, tcn_num_channels=tcn_num_channels, use_depthwise=False
        )

        model_dw = create_model(
            "tiny_conv", num_classes=num_classes, tcn_num_channels=tcn_num_channels, use_depthwise=True
        )

        params_std = sum(p.numel() for p in model_std.parameters())
        params_dw = sum(p.numel() for p in model_dw.parameters())

        reduction = (params_std - params_dw) / params_std

        print(f"\nStandard params: {params_std}")
        print(f"Depthwise params: {params_dw}")
        print(f"Reduction: {reduction:.2%}")

        # Target is ~70% reduction for typical configurations
        # We'll check if it's at least 50% for this specific config
        assert params_dw < params_std
        assert reduction > 0.5, f"Expected >50% reduction, got {reduction:.2%}"

    @pytest.mark.unit
    def test_tiny_conv_v2_qat_support(self):
        """Test that TinyConv V2 still supports QAT stubs"""
        model_dw = create_model("tiny_conv", num_classes=2, use_depthwise=True)
        assert hasattr(model_dw, "quant")
        assert hasattr(model_dw, "dequant")

        # Verify forward pass goes through quant/dequant
        # We can't easily check internal calls without mocking,
        # but we can ensure it doesn't crash.
        inputs = torch.randn(2, 1, 64, 50)
        outputs = model_dw(inputs)
        assert outputs.shape == (2, 2)
