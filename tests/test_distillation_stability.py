import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

# Add project root to path
sys.path.append(os.getcwd())

from src.config.defaults import WakewordConfig
from src.training.distillation_trainer import DistillationTrainer


class TestDistillationStability(unittest.TestCase):
    def setUp(self):
        self.config = WakewordConfig()
        self.config.distillation.enabled = True
        self.config.model.num_classes = 2

        self.student_model = MagicMock(spec=torch.nn.Module)
        self.student_model.return_value = torch.randn(2, 2)
        self.student_model.to.return_value = self.student_model

        self.optimizer = MagicMock()
        self.scheduler = MagicMock()
        self.train_loader = MagicMock()
        self.val_loader = MagicMock()
        self.checkpoint_manager = MagicMock()
        self.checkpoint_manager.checkpoint_dir = "test_checkpoints"

    @patch("src.training.distillation_trainer.Wav2VecWakeword")
    @patch("src.training.trainer.create_loss_function")
    @patch("src.training.trainer.AudioProcessor")
    @patch("src.training.trainer.create_optimizer_and_scheduler")
    def test_nan_logits_handling(self, mock_create_opt, mock_audio_proc, mock_create_loss, mock_wav2vec):
        mock_create_opt.return_value = (self.optimizer, self.scheduler)

        # Mock teacher returning NaN logits
        mock_teacher = MagicMock()
        mock_teacher.to.return_value = mock_teacher
        # Create logits with NaN
        nan_logits = torch.randn(2, 2)
        nan_logits[0, 0] = float("nan")
        mock_teacher.return_value = nan_logits
        mock_wav2vec.return_value = mock_teacher

        # Mock Criterion (Student Loss)
        mock_criterion = MagicMock()
        student_loss_val = torch.tensor(0.5)
        mock_criterion.return_value = student_loss_val
        mock_criterion.to.return_value = mock_criterion
        mock_create_loss.return_value = mock_criterion

        trainer = DistillationTrainer(
            model=self.student_model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=self.config,
            checkpoint_manager=self.checkpoint_manager,
            device="cpu",
        )

        outputs = torch.randn(2, 2)
        targets = torch.tensor([0, 1])
        raw_audio = torch.randn(2, 16000)

        # Compute loss
        loss = trainer.compute_loss(outputs, targets, inputs=raw_audio)

        # Should return ONLY student loss (0.5), skipping distillation
        self.assertTrue(torch.equal(loss, student_loss_val))

    @patch("src.training.distillation_trainer.Wav2VecWakeword")
    @patch("src.training.trainer.create_loss_function")
    @patch("src.training.trainer.AudioProcessor")
    @patch("src.training.trainer.create_optimizer_and_scheduler")
    def test_temperature_clamping(self, mock_create_opt, mock_audio_proc, mock_create_loss, mock_wav2vec):
        mock_create_opt.return_value = (self.optimizer, self.scheduler)

        # Bypass config property setter validation to test runtime clamping
        self.config.distillation._temperature = 0.1

        mock_teacher = MagicMock()
        mock_teacher.to.return_value = mock_teacher
        mock_teacher.return_value = torch.randn(2, 2)
        mock_wav2vec.return_value = mock_teacher

        mock_criterion = MagicMock()
        mock_criterion.return_value = torch.tensor(1.0)
        mock_criterion.to.return_value = mock_criterion
        mock_create_loss.return_value = mock_criterion

        trainer = DistillationTrainer(
            model=self.student_model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=self.config,
            checkpoint_manager=self.checkpoint_manager,
            device="cpu",
        )

        outputs = torch.randn(2, 2)
        targets = torch.tensor([0, 1])
        raw_audio = torch.randn(2, 16000)

        # This shouldn't crash
        loss = trainer.compute_loss(outputs, targets, inputs=raw_audio)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))


if __name__ == "__main__":
    unittest.main()
