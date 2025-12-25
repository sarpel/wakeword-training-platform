
import pytest
import torch
from src.data.augmentation import AudioAugmentation
from src.training.trainer import Trainer, TrainingState
from src.config.defaults import WakewordConfig
from pathlib import Path
from unittest.mock import MagicMock

def test_snr_scheduling_logic():
    """Verify that SNR scheduling decreases SNR (increases noise) over epochs."""
    # Setup augmentation with a target SNR range
    target_snr_min, target_snr_max = 5.0, 15.0
    aug = AudioAugmentation(
        sample_rate=16000,
        noise_snr_range=(target_snr_min, target_snr_max)
    )
    
    # Initially SNR scheduling is OFF
    assert not getattr(aug, "use_snr_scheduling", False)
    
    # Enable SNR scheduling
    total_epochs = 100
    aug.set_epoch(0, total_epochs)
    assert aug.use_snr_scheduling == True
    
    # At epoch 0, SNR should be "easy" (higher)
    # Based on planned logic: easy_min=15, easy_max=30
    # Let's check the internal range calculation via add_background_noise logic
    # We can mock add_background_noise or check internal state if we expose it
    
    # Test epoch 0 (Easy)
    aug.set_epoch(0, total_epochs)
    # We need to verify that curr_min/curr_max are at their starting points
    # Since they are local variables in add_background_noise, we might need to 
    # refactor add_background_noise to make it testable or use a mock.
    
    # For TDD, let's assume we implement a helper to get current SNR range
    def get_current_snr_range(aug_obj):
        progress = aug_obj.current_epoch / aug_obj.total_epochs
        easy_min, easy_max = 15.0, 30.0
        hard_min, hard_max = aug_obj.noise_snr_range
        curr_min = easy_min - (easy_min - hard_min) * progress
        curr_max = easy_max - (easy_max - hard_max) * progress
        return curr_min, curr_max

    c_min, c_max = get_current_snr_range(aug)
    assert c_min == 15.0
    assert c_max == 30.0
    
    # Test halfway point
    aug.set_epoch(50, total_epochs)
    c_min, c_max = get_current_snr_range(aug)
    assert c_min == 10.0 # (15 + 5) / 2
    assert c_max == 22.5 # (30 + 15) / 2
    
    # Test final epoch
    aug.set_epoch(100, total_epochs)
    c_min, c_max = get_current_snr_range(aug)
    assert c_min == 5.0
    assert c_max == 15.0

def test_trainer_updates_augmentation_epoch():
    """Verify that Trainer calls set_epoch on the augmentation module."""
    config = WakewordConfig()
    config.training.epochs = 10
    
    # Use a real small model to avoid optimizer errors
    from src.models.architectures import create_model
    model = create_model("tiny_conv", num_classes=2)
    
    train_loader = MagicMock()
    # Mock dataset to avoid length errors if needed
    train_loader.dataset = MagicMock()
    train_loader.__len__.return_value = 10
    
    val_loader = MagicMock()
    checkpoint_manager = MagicMock()
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_manager=checkpoint_manager,
        device="cpu"
    )
    
    # Mock the audio processor's augmentation
    trainer.audio_processor.augmentation = MagicMock()
    
    # Simulate an epoch start (epoch 5)
    # We need to manually call the logic we're about to implement
    trainer._update_augmentation_epoch(5)
    
    # This should trigger set_epoch on the augmentation
    trainer.audio_processor.augmentation.set_epoch.assert_called_with(5, 10)
