import pytest
from src.config.defaults import DistillationConfig, WakewordConfig
import tempfile
import yaml
import os

def test_default_values():
    """Test default configuration values"""
    config = DistillationConfig()
    assert config.enabled is False
    assert config.temperature == 2.0
    assert config.alpha == 0.5
    assert config.teacher_architecture == "wav2vec2"
    assert config.teacher_on_cpu is False
    assert config.teacher_mixed_precision is True
    assert config.log_memory_usage is False

def test_init_values():
    """Test initialization with values"""
    config = DistillationConfig(enabled=True, alpha=0.8, temperature=5.0)
    assert config.enabled is True
    assert config.alpha == 0.8
    assert config.temperature == 5.0

def test_alpha_validation():
    """Test alpha parameter validation"""
    # Valid values
    DistillationConfig(alpha=0.0)
    DistillationConfig(alpha=0.5)
    DistillationConfig(alpha=1.0)

    # Invalid values (Validation on init)
    with pytest.raises(ValueError, match="alpha must be in range"):
        DistillationConfig(alpha=-0.1)

    with pytest.raises(ValueError, match="alpha must be in range"):
        DistillationConfig(alpha=1.5)

def test_temperature_validation():
    """Test temperature parameter validation"""
    # Valid values
    DistillationConfig(temperature=1.0)
    DistillationConfig(temperature=5.0)
    DistillationConfig(temperature=10.0)

    # Invalid values (Validation on init)
    with pytest.raises(ValueError, match="temperature must be in range"):
        DistillationConfig(temperature=0.5)

    with pytest.raises(ValueError, match="temperature must be in range"):
        DistillationConfig(temperature=15.0)

def test_teacher_architecture_validation():
    """Test teacher architecture validation"""
    # Valid
    DistillationConfig(teacher_architecture="wav2vec2")

    # Invalid (Validation on init)
    with pytest.raises(ValueError, match="teacher_architecture must be one of"):
        DistillationConfig(teacher_architecture="unknown_model")

def test_config_to_dict():
    """Test configuration serialization"""
    config = DistillationConfig()
    config.enabled = True
    config.alpha = 0.7

    config_dict = config.to_dict()

    assert config_dict["enabled"] is True
    assert config_dict["alpha"] == 0.7
    assert "temperature" in config_dict
    assert "teacher_architecture" in config_dict
    assert "teacher_on_cpu" in config_dict
    assert "teacher_mixed_precision" in config_dict
    assert "log_memory_usage" in config_dict

def test_yaml_roundtrip():
    """Test saving and loading from YAML"""
    config = WakewordConfig()
    config.distillation.enabled = True
    config.distillation.alpha = 0.8
    config.distillation.temperature = 3.5

    # Save to YAML
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config.to_dict(), f)
        yaml_path = f.name

    try:
        # Load from YAML
        config_loaded = WakewordConfig.load(yaml_path)

        assert config_loaded.distillation.enabled is True
        assert config_loaded.distillation.alpha == 0.8
        assert config_loaded.distillation.temperature == 3.5
    finally:
        if os.path.exists(yaml_path):
            os.unlink(yaml_path)
