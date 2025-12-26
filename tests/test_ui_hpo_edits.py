from unittest.mock import MagicMock

import gradio as gr
import pandas as pd
import pytest

from src.config.defaults import WakewordConfig
from src.ui.panel_training import apply_best_params, save_best_profile


def test_apply_best_params_with_edits():
    """Test applying parameters from edited DataFrame."""
    config = WakewordConfig()
    config.training.learning_rate = 0.001
    config.training.batch_size = 32
    state = {"config": config, "best_hpo_params": {}}

    # Create DataFrame mimicking UI table
    data = {"Parameter": ["learning_rate", "batch_size", "new_param"], "Value": ["0.005", "64", "ignored"]}
    df = pd.DataFrame(data)

    # Apply
    msg = apply_best_params(state, df)

    # Verify config updated
    assert config.training.learning_rate == 0.005
    assert config.training.batch_size == 64
    assert "Applied" in msg


def test_save_best_profile():
    """Test saving profile with edited parameters."""
    config = WakewordConfig()
    state = {"config": config, "best_hpo_params": {}}

    # Create DataFrame
    data = {"Parameter": ["dropout"], "Value": ["0.5"]}
    df = pd.DataFrame(data)

    # Save (Mocking file I/O implicitly by checking config state before save call,
    # but actual save writes to disk. In unit test we should probably mock config.save)
    # However, defaults.py save method writes to disk.
    # Let's mock config.save to avoid file creation.
    config.save = MagicMock()

    msg = save_best_profile(state, df)

    assert config.model.dropout == 0.5
    assert config.save.called
    assert "Saved profile" in msg


if __name__ == "__main__":
    test_apply_best_params_with_edits()
    test_save_best_profile()
