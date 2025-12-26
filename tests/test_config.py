"""
Unit Tests for Configuration Module
Tests configuration loading, validation, and serialization
"""
from pathlib import Path

import pytest

from src.config.defaults import (
    AugmentationConfig,
    DataConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
    WakewordConfig,
)


class TestDefaultConfig:
    """Test default configuration values"""

    @pytest.mark.unit
    def test_default_config_creation(self, default_config):
        """Test default config can be created"""
        assert default_config is not None
        assert hasattr(default_config, "data")
        assert hasattr(default_config, "training")
        assert hasattr(default_config, "model")
        assert hasattr(default_config, "augmentation")
        assert hasattr(default_config, "optimizer")

    @pytest.mark.unit
    def test_data_config_defaults(self, default_config):
        """Test data configuration defaults"""
        data = default_config.data

        assert data.sample_rate == 16000
        assert data.audio_duration == 1.5
        assert data.n_mels == 64
        assert data.feature_type == "mel"

    @pytest.mark.unit
    def test_training_config_defaults(self):
        """Verify training configuration default values"""
        training = TrainingConfig()
        assert training.batch_size == 64
        assert training.epochs == 80
        assert training.learning_rate == 5e-4
        assert training.early_stopping_patience == 15
        assert training.num_workers == 16

    @pytest.mark.unit
    def test_model_config_defaults(self, default_config):
        """Test model configuration defaults"""
        model = default_config.model

        assert model.architecture == "resnet18"
        assert model.num_classes == 2
        assert model.dropout == 0.3

    @pytest.mark.unit
    def test_config_to_dict(self, default_config):
        """Test config serialization to dict"""
        data_dict = default_config.data.to_dict()

        assert isinstance(data_dict, dict)
        assert "sample_rate" in data_dict
        assert data_dict["sample_rate"] == 16000


class TestConfigSerialization:
    """Test configuration save/load"""

    @pytest.mark.unit
    def test_config_yaml_roundtrip(self, default_config, tmp_path):
        """Test config can be saved and loaded from YAML"""
        import yaml

        yaml_path = tmp_path / "config.yaml"

        # Save
        config_dict = {
            "data": default_config.data.to_dict(),
            "training": default_config.training.to_dict(),
            "model": default_config.model.to_dict(),
        }
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f)

        # Load
        with open(yaml_path, "r") as f:
            loaded = yaml.safe_load(f)

        assert loaded["data"]["sample_rate"] == 16000
        assert loaded["training"]["batch_size"] == 64


class TestAugmentationConfig:
    """Test augmentation configuration"""

    @pytest.mark.unit
    def test_augmentation_defaults(self, default_config):
        """Test augmentation defaults are sensible"""
        aug = default_config.augmentation

        assert aug.time_stretch_min < 1.0 < aug.time_stretch_max
        assert aug.pitch_shift_min < 0 < aug.pitch_shift_max
        assert 0.0 <= aug.background_noise_prob <= 1.0
        assert aug.noise_snr_min < aug.noise_snr_max

    @pytest.mark.unit
    def test_spec_augment_defaults(self, default_config):
        """Test SpecAugment configuration"""
        aug = default_config.augmentation

        assert aug.use_spec_augment is True
        assert aug.freq_mask_param > 0
        assert aug.time_mask_param > 0
        assert aug.n_freq_masks > 0
        assert aug.n_time_masks > 0


class TestOptimizerConfig:
    """Test optimizer configuration"""

    @pytest.mark.unit
    def test_optimizer_defaults(self, default_config):
        """Test optimizer defaults"""
        opt = default_config.optimizer

        assert opt.optimizer in ["adam", "adamw", "sgd"]
        assert opt.weight_decay >= 0
        assert opt.scheduler in ["cosine", "step", "plateau", "none"]

    @pytest.mark.unit
    def test_learning_rate_bounds(self, default_config):
        """Test learning rate is within sensible bounds"""
        training = default_config.training
        opt = default_config.optimizer

        assert 1e-6 <= training.learning_rate <= 1.0
        assert opt.min_lr < training.learning_rate
