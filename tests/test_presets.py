"""
Tests for Configuration Presets
"""
import pytest
from src.config.presets import get_preset

@pytest.mark.unit
def test_mcu_tiny_production_preset():
    """Verify the MCU Tiny Production preset has high-quality settings"""
    config = get_preset("MCU (ESP32-S3 No-PSRAM)")
    
    # Check updated resolution and duration
    assert config.data.n_mels == 64
    assert config.data.audio_duration == 1.5
    
    # Check updated capacity
    assert config.model.tcn_num_channels == [64, 64, 64, 64]
    
    # Check distillation enabled
    assert config.distillation.enabled is True
    assert config.distillation.teacher_architecture == "wav2vec2"
    
    # Check EMA enabled
    assert config.training.use_ema is True
    
    # Check size targets for S3
    assert config.size_targets.max_flash_kb == 256
    assert config.size_targets.max_ram_kb == 192

@pytest.mark.unit
def test_preset_names():
    """Verify all expected presets exist"""
    from src.config.presets import list_presets
    presets = list_presets()
    assert "MCU (ESP32-S3 No-PSRAM)" in presets
    assert "RPI (Raspberry Pi / Wyoming Satellite)" in presets
    assert "x86_64 (Desktop / Server)" in presets
