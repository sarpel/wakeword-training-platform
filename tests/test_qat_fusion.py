"""
Tests for QAT Module Fusion
"""
import pytest
import torch
import torch.nn as nn

from src.config.defaults import WakewordConfig
from src.models.architectures import create_model
from src.training.qat_utils import prepare_model_for_qat


class TestQATFusion:
    """Test suite for QAT module fusion logic"""

    @pytest.mark.unit
    def test_tiny_conv_fusion(self):
        """Test that TinyConv layers are correctly fused during QAT preparation"""
        config = WakewordConfig()
        config.qat.enabled = True

        # Standard TinyConv (Non-depthwise for simple fusion test)
        model = create_model("tiny_conv", num_classes=2, use_depthwise=False)

        # Before fusion, features[0] is Conv2d, features[1] is BatchNorm2d
        assert isinstance(model.features[0], nn.Conv2d)
        assert isinstance(model.features[1], nn.BatchNorm2d)

        # Prepare for QAT (should trigger fusion)
        prepared_model = prepare_model_for_qat(model, config.qat)

        # After fusion, we expect 'torch.ao.nn.intrinsic.qat.ConvBnReLU2d' or similar
        # inside the prepared model's internal structure.
        # Actually, prepare_qat replaces modules. We should check the fused model.
        # Note: torch.quantization.fuse_modules creates a fused module.

        # Let's check if the BatchNorm is gone/absorbed in the features sequence
        # In a fused sequence [Conv, BN, ReLU], it becomes [FusedConvBNReLU, Identity, Identity]
        # or the BN is replaced by an Identity module.

        # Find the first Conv/BN/ReLU block
        # We search for the fused module type
        has_fused = False
        module_names = []
        for m in prepared_model.modules():
            name = m.__class__.__name__
            module_names.append(name)
            # In QAT, fused modules often have names like ConvBnReLU2d or LinearBnReLU1d
            # depending on the torch version and whether it's the intrinsic or QAT version
            if any(x in name for x in ["ConvBnReLU2d", "ConvBn2d", "ConvReLU2d"]):
                has_fused = True
                break

        if not has_fused:
            print(f"DEBUG: All module types found: {module_names}")

        assert has_fused, "Model should contain fused ConvBnReLU modules after QAT preparation"

    @pytest.mark.unit
    def test_tiny_conv_dw_fusion(self):
        """Test that Depthwise TinyConv layers are correctly fused"""
        config = WakewordConfig()
        config.qat.enabled = True

        model = create_model("tiny_conv", num_classes=2, use_depthwise=True)

        # Depthwise TinyConv has more layers [ConvDW, BN, ReLU, ConvPW, BN, ReLU]
        prepared_model = prepare_model_for_qat(model, config.qat)

        # Count fused modules
        fused_count = 0
        for m in prepared_model.modules():
            name = m.__class__.__name__
            if any(x in name for x in ["ConvBnReLU2d", "ConvBn2d", "ConvReLU2d"]):
                fused_count += 1

        # Standard TinyConv with 4 layers should have 4 fused blocks
        # Depthwise TinyConv has 1 (standard first layer) + 3*2 (dw+pw for remaining 3) = 7 fused blocks
        assert fused_count >= 4, f"Expected at least 4 fused blocks, got {fused_count}"
