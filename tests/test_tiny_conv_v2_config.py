"""
Tests for TinyConv V2 Configuration
"""

import pytest

from src.config.defaults import ModelConfig


class TestTinyConvV2Config:
    """Test suite for TinyConv V2 configuration parameters"""

    @pytest.mark.unit
    def test_model_config_has_tiny_conv_use_depthwise(self):
        """Test that ModelConfig includes tiny_conv_use_depthwise with default False"""
        model_config = ModelConfig()
        assert hasattr(model_config, "tiny_conv_use_depthwise")
        assert model_config.tiny_conv_use_depthwise is False

    @pytest.mark.unit
    def test_model_config_serialization(self):
        """Test that tiny_conv_use_depthwise is serialized correctly"""
        model_config = ModelConfig()
        model_config.tiny_conv_use_depthwise = True

        config_dict = model_config.to_dict()
        assert config_dict["tiny_conv_use_depthwise"] is True
