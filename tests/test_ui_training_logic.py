from unittest.mock import MagicMock, patch

import pytest

from src.config.defaults import WakewordConfig
from src.ui.panel_training import start_training


def test_start_training_updates_loss_config():
    """Test that start_training updates config with UI values."""

    # Mock Config State
    config = WakewordConfig()
    config_state = {"config": config}

    # Mock other dependencies to avoid side effects
    with patch("src.ui.panel_training.training_state") as mock_state:
        mock_state.is_training = False

        # We expect it to return error because datasets/paths not found in test env
        # But we only care that it attempted to update config BEFORE failing or starting

        # Actually, start_training updates config inside the try block.
        # If we can mock `paths.SPLITS` etc to pass checks, we can reach the update.

        with patch("src.ui.panel_training.paths") as mock_paths:
            mock_paths.CHECKPOINTS.exists.return_value = True
            mock_paths.SPLITS.exists.return_value = True
            (mock_paths.SPLITS / "train.json").exists.return_value = True

            # We also need to mock load_dataset_splits or it will try to load data
            with patch("src.ui.panel_training.load_dataset_splits") as mock_load:
                mock_load.return_value = (MagicMock(), MagicMock(), MagicMock())

                # Mock DataLoader creation
                with patch("src.ui.panel_training.DataLoader"):
                    # Mock create_model
                    with patch("src.ui.panel_training.create_model"):
                        # Mock Trainer
                        with patch("src.ui.panel_training.Trainer"):
                            # Mock thread
                            with patch("threading.Thread"):
                                start_training(
                                    config_state=config_state,
                                    use_cmvn=False,
                                    use_ema=True,
                                    ema_decay=0.9995,
                                    use_balanced_sampler=False,
                                    sampler_ratio_pos=1,
                                    sampler_ratio_neg=1,
                                    sampler_ratio_hard=0,
                                    run_lr_finder=False,
                                    use_wandb=False,
                                    wandb_project="test",
                                    wandb_api_key="",
                                    resume_checkpoint=None,
                                )

                                # Check session overrides
                                assert config.training.use_ema == True
                                assert config.training.ema_decay == 0.9995
