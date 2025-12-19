import torch
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.training.distillation_trainer import DistillationTrainer
from src.config.defaults import WakewordConfig

class TestDistillationTrainer(unittest.TestCase):
    def setUp(self):
        # Create config with distillation enabled
        self.config = WakewordConfig()
        self.config.distillation.enabled = True
        self.config.distillation.teacher_architecture = "wav2vec2"
        self.config.model.num_classes = 2
        
        # Mock model, optimizer, etc.
        self.student_model = MagicMock(spec=torch.nn.Module)
        self.student_model.return_value = torch.randn(2, 2) # Batch size 2, 2 classes
        self.student_model.to.return_value = self.student_model
        
        self.optimizer = MagicMock()
        self.scheduler = MagicMock()
        
        # Mock loaders
        self.train_loader = MagicMock()
        self.val_loader = MagicMock()
        
        # Mock checkpoint manager
        self.checkpoint_manager = MagicMock()
        self.checkpoint_manager.checkpoint_dir = "test_checkpoints"

    @patch("src.training.distillation_trainer.Wav2VecWakeword")
    @patch("src.training.trainer.create_loss_function")
    @patch("src.training.trainer.AudioProcessor")
    @patch("src.training.trainer.create_optimizer_and_scheduler")
    def test_distillation_logic(self, mock_create_opt, mock_audio_proc, mock_create_loss, mock_wav2vec):
        # Setup mocks
        mock_create_opt.return_value = (self.optimizer, self.scheduler)
        
        # Mock Teacher
        mock_teacher_instance = MagicMock()
        mock_teacher_instance.to.return_value = mock_teacher_instance
        mock_teacher_instance.eval.return_value = None
        # Teacher returns logits
        mock_teacher_instance.return_value = torch.randn(2, 2) 
        mock_wav2vec.return_value = mock_teacher_instance

        # Mock Criterion to return a Tensor
        mock_criterion = MagicMock()
        mock_criterion.return_value = torch.tensor(1.0) # Dummy loss
        mock_criterion.to.return_value = mock_criterion
        mock_create_loss.return_value = mock_criterion
        
        # Initialize Trainer
        trainer = DistillationTrainer(
            model=self.student_model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=self.config,
            checkpoint_manager=self.checkpoint_manager,
            device="cpu" # Use CPU for test
        )
        
        # Verify teacher initialized
        mock_wav2vec.assert_called_once()
        self.assertIsNotNone(trainer.teacher)
        
        # Test compute_loss with RAW AUDIO (2D tensor)
        # Batch size 2, 16000 samples
        raw_inputs = torch.randn(2, 16000)
        outputs = torch.randn(2, 2)
        targets = torch.randint(0, 2, (2,))
        
        # Call compute_loss (with optional is_hard_negative parameter)
        loss = trainer.compute_loss(outputs, targets, inputs=raw_inputs, is_hard_negative=None)

        # Verify teacher was called
        mock_teacher_instance.assert_called_once_with(raw_inputs)
        
        # Verify loss is a tensor
        self.assertIsInstance(loss, torch.Tensor)
        print("Distillation logic verified: Teacher called with raw inputs.")

    @patch("src.training.distillation_trainer.Wav2VecWakeword")
    @patch("src.training.trainer.create_loss_function")
    @patch("src.training.trainer.AudioProcessor")
    @patch("src.training.trainer.create_optimizer_and_scheduler")
    def test_distillation_skip_on_spectrograms(self, mock_create_opt, mock_audio_proc, mock_create_loss, mock_wav2vec):
        # Setup mocks
        mock_create_opt.return_value = (self.optimizer, self.scheduler)
        
        # Mock Teacher
        mock_teacher_instance = MagicMock()
        mock_wav2vec.return_value = mock_teacher_instance
        
        # Initialize Trainer
        trainer = DistillationTrainer(
            model=self.student_model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=self.config,
            checkpoint_manager=self.checkpoint_manager,
            device="cpu"
        )
        
        # Test compute_loss with SPECTROGRAMS (4D tensor)
        # Batch, Channel, Freq, Time
        spectrogram_inputs = torch.randn(2, 1, 64, 100)
        outputs = torch.randn(2, 2)
        targets = torch.randint(0, 2, (2,))
        
        # Call compute_loss (with optional is_hard_negative parameter)
        loss = trainer.compute_loss(outputs, targets, inputs=spectrogram_inputs, is_hard_negative=None)

        # Verify teacher was NOT called
        mock_teacher_instance.assert_not_called()
        print("Distillation skip verified: Teacher NOT called with spectrogram inputs.")

if __name__ == "__main__":
    unittest.main()
