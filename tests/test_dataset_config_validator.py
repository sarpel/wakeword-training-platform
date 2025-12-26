"""
Tests for DatasetConfigValidator
"""
import json
from pathlib import Path

import numpy as np
import pytest

from src.config.dataset_config_validator import DatasetConfigValidator
from src.config.defaults import WakewordConfig


def test_validator_mismatch_shape(temp_data_dir):
    """Test detection of shape mismatch"""
    # Setup: Create a config with 16kHz, 64 mels, 1.5s
    config = WakewordConfig()
    config.data.sample_rate = 16000
    config.data.n_mels = 64
    config.data.audio_duration = 1.5

    # Create an NPY file with wrong shape (e.g., 40 mels)
    npy_dir = temp_data_dir / "npy"
    npy_path = npy_dir / "test_sample.npy"
    # Shape: (1, 40, 151) instead of (1, 64, 151)
    np.save(npy_path, np.zeros((1, 40, 151)))

    validator = DatasetConfigValidator()
    mismatches = validator.validate_dataset_features(config, npy_dir)

    assert len(mismatches) > 0
    assert mismatches[0]["actual_shape"] == (1, 40, 151)
    assert mismatches[0]["expected_shape"] == (1, 64, 151)


def test_validator_match(temp_data_dir):
    """Test detection of correct shape"""
    config = WakewordConfig()
    config.data.sample_rate = 16000
    config.data.n_mels = 64
    config.data.audio_duration = 1.5

    npy_dir = temp_data_dir / "npy"
    npy_path = npy_dir / "test_sample.npy"
    # Shape: (1, 64, 151)
    np.save(npy_path, np.zeros((1, 64, 151)))

    validator = DatasetConfigValidator()
    mismatches = validator.validate_dataset_features(config, npy_dir)

    assert len(mismatches) == 0


def test_validator_mismatch_duration(temp_data_dir):
    """Test detection of duration mismatch (time steps)"""
    config = WakewordConfig()
    config.data.sample_rate = 16000
    config.data.n_mels = 64
    config.data.audio_duration = 2.0  # Changed from default 1.5

    npy_dir = temp_data_dir / "npy"
    npy_path = npy_dir / "test_sample.npy"
    # 1.5s @ 16kHz, hop 160 -> 151 steps
    # 2.0s @ 16kHz, hop 160 -> 201 steps
    np.save(npy_path, np.zeros((1, 64, 151)))

    validator = DatasetConfigValidator()
    mismatches = validator.validate_dataset_features(config, npy_dir)

    assert len(mismatches) > 0
    assert mismatches[0]["expected_shape"] == (1, 64, 201)
