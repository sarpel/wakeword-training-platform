"""
Tests for newly added configuration parameters (CMVN, Streaming, Size Targets, Calibration).
"""
import pytest

from src.config.defaults import get_default_config
from src.config.presets import get_preset, list_presets
from src.config.pydantic_validator import validate_config_with_pydantic


class TestNewConfigParams:
    """Test the newly added configuration parameters"""

    def test_default_config_has_new_sections(self):
        """Test that default config includes the new sections with default values"""
        config = get_default_config()

        assert hasattr(config, "cmvn")
        assert hasattr(config, "streaming")
        assert hasattr(config, "size_targets")
        assert hasattr(config, "calibration")

        assert config.cmvn.enabled is True
        assert config.streaming.hysteresis_high == 0.7
        assert config.size_targets.max_flash_kb == 0
        assert config.calibration.num_samples == 100

    def test_presets_have_custom_new_params(self):
        """Test that production presets have custom values for new parameters"""
        # Test MCU Tiny preset
        mcu_preset = get_preset("MCU (ESP32-S3 No-PSRAM)")

        assert mcu_preset.size_targets.max_flash_kb == 100
        assert mcu_preset.size_targets.max_ram_kb == 80
        assert mcu_preset.streaming.hysteresis_high == 0.8
        assert mcu_preset.calibration.num_samples == 300

        # Test x86 Ultimate preset
        x86_preset = get_preset("x86_64 (Desktop / Server)")
        assert x86_preset.streaming.smoothing_window == 10
        assert x86_preset.calibration.num_samples == 500

    def test_pydantic_validation_of_new_params(self):
        """Test that Pydantic correctly validates the new parameters"""
        config = get_default_config()
        config_dict = config.to_dict()

        # Valid case
        is_valid, errors = validate_config_with_pydantic(config_dict)
        assert is_valid is True
        assert len(errors) == 0

        # Invalid CMVN path (not caught by pydantic as it's just a string, but let's test a numeric constraint)
        config_dict["streaming"]["hysteresis_high"] = 1.5  # Should be <= 1.0
        is_valid, errors = validate_config_with_pydantic(config_dict)
        assert is_valid is False
        assert any("hysteresis_high" in str(e["loc"]) for e in errors)

        # Invalid Calibration samples
        config_dict["streaming"]["hysteresis_high"] = 0.7  # fix previous
        config_dict["calibration"]["num_samples"] = 5  # Should be >= 10
        is_valid, errors = validate_config_with_pydantic(config_dict)
        assert is_valid is False
        assert any("num_samples" in str(e["loc"]) for e in errors)

    def test_serialization_roundtrip(self, tmp_path):
        """Test that the new parameters survive YAML serialization"""
        config = get_preset("MCU (ESP32-S3 No-PSRAM)")
        yaml_path = tmp_path / "mcu_config.yaml"

        config.save(yaml_path)

        from src.config.defaults import WakewordConfig

        loaded_config = WakewordConfig.load(yaml_path)

        assert loaded_config.size_targets.max_flash_kb == 100
        assert loaded_config.streaming.hysteresis_high == 0.8
        assert loaded_config.calibration.num_samples == 300
        assert loaded_config.cmvn.enabled is True
