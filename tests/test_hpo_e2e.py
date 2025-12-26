from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.config.defaults import WakewordConfig
from src.training.hpo_results import HPOResult
from src.ui.panel_training import apply_best_params, save_best_profile, start_hpo, training_state


def test_e2e_hpo_to_profile():
    """
    Simulate full flow: HPO run -> Result capture -> Table edit -> Save profile.
    """
    config = WakewordConfig()
    state = {"config": config}

    # 1. Mock HPO Result
    best_params = {"learning_rate": 0.0005, "batch_size": 64, "dropout": 0.4, "pitch_shift_range": 2}
    mock_result = HPOResult(study_name="e2e_test", best_value=0.9, best_params=best_params, n_trials=10, duration=5.0)

    # 2. Simulate start_hpo finishing (using mock)
    # In reality this happens in a thread, here we simulate the result capture
    state["best_hpo_params"] = mock_result.best_params
    training_state.hpo_result = mock_result

    # 3. Simulate UI Table Edit
    # User changes dropout from 0.4 to 0.5 in the table
    edited_df = pd.DataFrame(
        [
            {"Parameter": "learning_rate", "Value": 0.0005},
            {"Parameter": "batch_size", "Value": 64},
            {"Parameter": "dropout", "Value": 0.5},
            {"Parameter": "pitch_shift_range", "Value": 2},
        ]
    )

    # 4. Apply Edits
    apply_msg = apply_best_params(state, edited_df)
    assert "Applied 4 parameters" in apply_msg
    assert config.model.dropout == 0.5
    assert config.augmentation.pitch_shift_max == 2

    # 5. Save Profile
    config.save = MagicMock()
    save_msg = save_best_profile(state, edited_df)
    assert "Saved profile" in save_msg
    assert config.save.called

    print("\nE2E Flow Test Passed!")


if __name__ == "__main__":
    test_e2e_hpo_to_profile()
