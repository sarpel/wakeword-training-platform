import pytest
from pathlib import Path
from src.data.dataset import WakewordDataset
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_manifest(tmp_path):
    manifest_path = tmp_path / "train.json"
    import json
    data = {
        "files": [],
        "categories": ["positive", "negative"]
    }
    with open(manifest_path, "w") as f:
        json.dump(data, f)
    return manifest_path

def test_dataset_augmentation_config_pass_through(mock_manifest):
    """Test that augmentation config is correctly passed to AudioAugmentation."""
    
    aug_config = {
        "time_shift_prob": 0.8,
        "time_shift_range_ms": (-50, 50)
    }
    
    # We need to mock AudioAugmentation to check its init args
    # OR we can inspect the instance if it's stored.
    # WakewordDataset stores it in self.augmentation
    
    ds = WakewordDataset(
        manifest_path=mock_manifest,
        augment=True,
        augmentation_config=aug_config,
        return_raw_audio=False # Ensure Augmentation is initialized
    )
    
    assert ds.augmentation is not None
    # Use getattr to avoid attribute error if they are not yet implemented in dataset.py logic
    # But since we are testing IF they are passed, we expect them to be set in the Augmentation object
    # The Augmentation object logic sets them from its init args.
    
    # If dataset.py DOES NOT pass them, they will have default values from augmentation.py
    # Default time_shift_prob is 0.5 in augmentation.py (from my read)
    
    assert ds.augmentation.time_shift_prob == 0.8
    assert ds.augmentation.time_shift_range_ms == (-50, 50)
