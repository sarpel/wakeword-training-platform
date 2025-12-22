import torch
import torch.nn as nn
import pytest
from src.models.losses import create_loss_function, FocalLoss, LabelSmoothingCrossEntropy

@pytest.fixture
def device():
    return "cpu"

def test_weighted_cross_entropy_respects_weights(device):
    """Test that LabelSmoothingCrossEntropy respects class weights."""
    # 2 classes, weight class 1 higher
    weights = torch.tensor([1.0, 10.0]).to(device)
    loss_fn = LabelSmoothingCrossEntropy(smoothing=0.0, weight=weights, reduction='none')
    
    # Prediction: Perfect for class 0, Perfect for class 1
    # logits: [10, -10] for class 0, [-10, 10] for class 1
    # We want to check bad predictions to see impact of weights
    
    # Case 1: Wrong prediction for class 0 (Weight 1.0)
    # Target 0, Pred [ -10, 10 ] (predicts 1)
    pred_bad_0 = torch.tensor([[-10.0, 10.0]]).to(device)
    target_0 = torch.tensor([0]).to(device)
    loss_0 = loss_fn(pred_bad_0, target_0)
    
    # Case 2: Wrong prediction for class 1 (Weight 10.0)
    # Target 1, Pred [ 10, -10 ] (predicts 0)
    pred_bad_1 = torch.tensor([[10.0, -10.0]]).to(device)
    target_1 = torch.tensor([1]).to(device)
    loss_1 = loss_fn(pred_bad_1, target_1)
    
    # Loss 1 should be approx 10x Loss 0
    ratio = loss_1.item() / loss_0.item()
    assert 9.0 < ratio < 11.0, f"Expected ratio ~10, got {ratio}"

def test_focal_loss_alpha_vs_weight_precedence(device):
    """Test that weight takes precedence over alpha in FocalLoss if both provided (as per warning logic)."""
    weights = torch.tensor([1.0, 5.0]).to(device)
    # Init with both
    loss_fn = FocalLoss(alpha=0.25, gamma=0.0, weight=weights, reduction='none')
    
    assert loss_fn.alpha is None # The init logic sets alpha to None if weight is provided
    
    # Check if weights are applied
    # Gamma 0 => approx Cross Entropy
    
    pred_bad_0 = torch.tensor([[-10.0, 10.0]]).to(device)
    target_0 = torch.tensor([0]).to(device)
    loss_0 = loss_fn(pred_bad_0, target_0)
    
    pred_bad_1 = torch.tensor([[10.0, -10.0]]).to(device)
    target_1 = torch.tensor([1]).to(device)
    loss_1 = loss_fn(pred_bad_1, target_1)
    
    ratio = loss_1.item() / loss_0.item()
    assert 4.5 < ratio < 5.5, f"Expected ratio ~5, got {ratio}"

def test_factory_supports_class_weights(device):
    """Test create_loss_function accepts and uses class_weights."""
    weights = torch.tensor([0.1, 0.9]).to(device)
    loss_fn = create_loss_function("cross_entropy", class_weights=weights, device=device)
    
    assert isinstance(loss_fn, (LabelSmoothingCrossEntropy, nn.CrossEntropyLoss))
    # If using LabelSmoothingCrossEntropy wrapper
    if hasattr(loss_fn, 'weight'):
        assert torch.allclose(loss_fn.weight, weights)
