import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock wandb before importing WandbCallback to avoid ImportError if wandb is not installed
mock_wandb = MagicMock()
sys.modules['wandb'] = mock_wandb

from src.training.wandb_callback import WandbCallback
from src.training.metrics import MetricResults

class TestWandbCallback(unittest.TestCase):
    def setUp(self):
        # Reset the mock before each test
        mock_wandb.reset_mock()
        mock_wandb.run = None

    def test_init_uses_recommended_parameters(self):
        """Test that wandb.init is called with recommended parameters."""
        project_name = "test_project"
        config = {"learning_rate": 0.001, "batch_size": 32}
        
        # Initialize the callback
        callback = WandbCallback(project_name, config)
        
        # Verify wandb.init was called with reinit="finish_previous" instead of reinit=True
        mock_wandb.init.assert_called_once_with(
            project=project_name,
            config=config,
            reinit="finish_previous"
        )

    def test_init_finishes_existing_run(self):
        """Test that an existing run is finished before starting a new one."""
        mock_wandb.run = MagicMock()
        project_name = "test_project"
        config = {}
        
        WandbCallback(project_name, config)
        
        # Verify wandb.finish() was called
        mock_wandb.finish.assert_called()

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
            negative_samples=95
        )
        
        callback.on_epoch_end(epoch=0, train_loss=0.5, val_loss=0.4, val_metrics=mock_metrics)
        
        # Verify wandb.log was called with expected dictionary
        mock_wandb.log.assert_called()
        args, kwargs = mock_wandb.log.call_args
        log_dict = args[0]
        
        self.assertEqual(log_dict["epoch"], 1)
        self.assertEqual(log_dict["train_loss"], 0.5)
        self.assertEqual(log_dict["val_loss"], 0.4)
        self.assertEqual(log_dict["val_accuracy"], 0.9)
        self.assertEqual(log_dict["val_f1"], 0.82)

    def test_on_batch_end_logs_metrics(self):
        """Test that metrics are logged correctly at the end of a batch."""
        project_name = "test_project"
        config = {}
        callback = WandbCallback(project_name, config)
        
        callback.on_batch_end(batch_idx=0, loss=0.5, acc=0.8, step=10)
        
        # Verify wandb.log was called with expected dictionary and step
        mock_wandb.log.assert_called_with({"batch_loss": 0.5, "batch_acc": 0.8}, step=10)

    def test_on_batch_end_without_step(self):
        """Test that metrics are logged correctly at the end of a batch without step."""
        project_name = "test_project"
        config = {}
        callback = WandbCallback(project_name, config)
        
        callback.on_batch_end(batch_idx=0, loss=0.5, acc=0.8)
        
        # Verify wandb.log was called with expected dictionary
        mock_wandb.log.assert_called_with({"batch_loss": 0.5, "batch_acc": 0.8})

    def test_on_train_end_finishes_run(self):
        """Test that wandb.finish is called when training ends."""
        project_name = "test_project"
        config = {}
        callback = WandbCallback(project_name, config)
        
        callback.on_train_end()
        
        # Verify wandb.finish was called
        mock_wandb.finish.assert_called()

if __name__ == '__main__':
    unittest.main()
