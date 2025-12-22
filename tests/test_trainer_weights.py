import pytest
import torch
from unittest.mock import MagicMock
from src.training.trainer import Trainer
from src.config.defaults import WakewordConfig
from src.training.checkpoint_manager import CheckpointManager

def test_trainer_initializes_with_class_weights():
    # Mock config
    config = WakewordConfig()
    config.training.use_ema = False 
    
    # Mock dataset
    mock_dataset = MagicMock()
    mock_dataset.get_class_weights.return_value = torch.tensor([0.1, 0.9])
    
    # Mock loader
    mock_loader = MagicMock()
    mock_loader.dataset = mock_dataset
    mock_loader.__len__.return_value = 10
    
    # Mock model
    model = MagicMock()
    model.to.return_value = model
    
    # Mock checkpoint manager
    ckpt = MagicMock(spec=CheckpointManager)
    ckpt.checkpoint_dir = "test_dir"
    
    # Init Trainer (CPU)
    # We need to mock create_optimizer_and_scheduler inside Trainer or mock the model parameters
    model.parameters.return_value = [torch.randn(1, requires_grad=True)]
    
    trainer = Trainer(
        model=model,
        train_loader=mock_loader,
        val_loader=mock_loader,
        config=config,
        checkpoint_manager=ckpt,
        device="cpu"
    )
    
    # Access criterion
    criterion = trainer.criterion
    
    # Check if weight attribute exists and matches
    # Default config usually creates LabelSmoothingCrossEntropy or CrossEntropy
    if hasattr(criterion, 'weight') and criterion.weight is not None:
        assert torch.allclose(criterion.weight, torch.tensor([0.1, 0.9]))
    else:
        # If it's None, it failed
        pytest.fail(f"Criterion weight is None or missing. Criterion type: {type(criterion)}")
