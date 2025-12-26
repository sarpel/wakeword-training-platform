import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.metrics import MetricResults
from src.training.wandb_callback import WandbCallback


class TestWandbCallback(unittest.TestCase):
    def setUp(self):
        # Create mocks for wandb and weave
        self.mock_wandb = MagicMock()
        self.mock_weave = MagicMock()

        # Patch them in the module where they are used
        self.wandb_patcher = patch("src.training.wandb_callback.wandb", self.mock_wandb)
        self.weave_patcher = patch("src.training.wandb_callback.weave", self.mock_weave)

        self.wandb_patcher.start()
        self.weave_patcher.start()

        self.mock_wandb.run = None

    def tearDown(self):
        self.wandb_patcher.stop()
        self.weave_patcher.stop()

    def test_init_uses_recommended_parameters(self):
        """Test that wandb.init is called with recommended parameters."""
        project_name = "test_project"
        config = {"learning_rate": 0.001, "batch_size": 32}

        # Initialize the callback
        callback = WandbCallback(project_name, config)

        # Verify wandb.init was called with reinit="finish_previous"
        self.mock_wandb.init.assert_called_once_with(project=project_name, config=config, reinit="finish_previous")

    def test_init_finishes_existing_run(self):
        """Test that an existing run is finished before starting a new one."""
        self.mock_wandb.run = MagicMock()
        project_name = "test_project"
        config = {}

        WandbCallback(project_name, config)

        # Verify wandb.finish() was called
        self.mock_wandb.finish.assert_called()

    def test_init_initializes_weave(self):
        """Test that weave.init is called."""
        project_name = "test_project"
        config = {}

        WandbCallback(project_name, config)

        self.mock_weave.init.assert_called_once_with(project_name)

    def test_on_epoch_end_logs_metrics(self):
        """Test that metrics are logged correctly at the end of an epoch."""
        project_name = "test_project"
        config = {}
        callback = WandbCallback(project_name, config)

        mock_metrics = MetricResults(
            accuracy=0.9,
            precision=0.85,
            recall=0.8,
            f1_score=0.82,
            fpr=0.05,
            fnr=0.2,
            true_positives=80,
            true_negatives=90,
            false_positives=5,
            false_negatives=20,
            total_samples=195,
            positive_samples=100,
            negative_samples=95,
            pauc=0.75,
            eer=0.1,
            fah=0.5,
        )

        callback.on_epoch_end(epoch=0, train_loss=0.5, val_loss=0.4, val_metrics=mock_metrics)

        # Verify wandb.log was called with expected dictionary
        self.mock_wandb.log.assert_called()
        args, kwargs = self.mock_wandb.log.call_args
        log_dict = args[0]

        self.assertEqual(log_dict["epoch"], 1)
        self.assertEqual(log_dict["train_loss"], 0.5)
        self.assertEqual(log_dict["val_loss"], 0.4)
        self.assertEqual(log_dict["val_accuracy"], 0.9)
        self.assertEqual(log_dict["val_f1"], 0.82)
        self.assertEqual(log_dict["val_eer"], 0.1)
        self.assertEqual(log_dict["val_fah"], 0.5)

    def test_on_batch_end_logs_metrics(self):
        """Test that metrics are logged correctly at the end of a batch."""
        project_name = "test_project"
        config = {}
        callback = WandbCallback(project_name, config)

        callback.on_batch_end(batch_idx=0, loss=0.5, acc=0.8, step=10)

        # Verify wandb.log was called with expected dictionary and step
        self.mock_wandb.log.assert_called_with({"batch_loss": 0.5, "batch_acc": 0.8}, step=10)

    def test_on_batch_end_without_step(self):
        """Test that metrics are logged correctly at the end of a batch without step."""
        project_name = "test_project"
        config = {}
        callback = WandbCallback(project_name, config)

        callback.on_batch_end(batch_idx=0, loss=0.5, acc=0.8)

        # Verify wandb.log was called with expected dictionary
        self.mock_wandb.log.assert_called_with({"batch_loss": 0.5, "batch_acc": 0.8})

    def test_on_train_end_finishes_run(self):
        """Test that wandb.finish is called when training ends."""
        project_name = "test_project"
        config = {}
        callback = WandbCallback(project_name, config)

        callback.on_train_end()

        # Verify wandb.finish was called
        self.mock_wandb.finish.assert_called()


if __name__ == "__main__":
    unittest.main()
