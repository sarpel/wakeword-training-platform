"""
Tests for loading HPO profiles
"""
import pytest
import json
from pathlib import Path
from src.config.defaults import WakewordConfig

@pytest.fixture
def mock_hpo_profile(tmp_path):
    """Create a mock HPO profile for testing"""
    profile_dir = tmp_path / "configs" / "profiles"
    profile_dir.mkdir(parents=True)
    
    profile_data = {
        "metadata": {"group": "Complete", "description": "Test HPO"},
        "parameters": {
            "training": {"learning_rate": 0.00123, "batch_size": 128},
            "model": {"dropout": 0.45},
            "augmentation": {"background_noise_prob": 0.77},
            "loss": {"loss_function": "focal_loss"}
        }
    }
    
    profile_path = profile_dir / "hpo_best_complete.json"
    with open(profile_path, "w") as f:
        json.dump(profile_data, f)
        
    return profile_path

@pytest.mark.unit
def test_load_latest_hpo_profile(mock_hpo_profile):
    """Verify that we can load the latest HPO profile into config"""
    config = WakewordConfig()
    
    # We'll need to pass the directory to the function for testing
    # Or mock the path in the function
    from src.config.defaults import load_latest_hpo_profile
    
    # Update the latest profile
    success = load_latest_hpo_profile(config, profile_dir=mock_hpo_profile.parent)
    
    assert success is True
    assert config.training.learning_rate == 0.00123
    assert config.training.batch_size == 128
    assert config.model.dropout == 0.45
    assert config.augmentation.background_noise_prob == 0.77
    assert config.loss.loss_function == "focal_loss"

@pytest.mark.unit
def test_load_latest_hpo_profile_not_found():
    """Verify handling when no HPO profile exists"""
    config = WakewordConfig()
    from src.config.defaults import load_latest_hpo_profile
    
    # Non-existent directory
    success = load_latest_hpo_profile(config, profile_dir=Path("non_existent_dir_12345"))
    assert success is False
