import torch
import pytest
from src.training.metrics import MetricsCalculator

def test_metrics_calculator_with_pauc():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calculator = MetricsCalculator(device=device)
    
    # Perfect predictions
    logits = torch.tensor([
        [1.0, -1.0], # Label 0
        [1.0, -1.0], # Label 0
        [-1.0, 1.0], # Label 1
        [-1.0, 1.0], # Label 1
    ]).to(device)
    labels = torch.tensor([0, 0, 1, 1]).to(device)
    
    metrics = calculator.calculate(logits, labels)
    # Check if pauc is in the result (it should be after my changes)
    assert hasattr(metrics, 'pauc')
    assert metrics.pauc > 0.9

def test_metrics_tracker_with_pauc():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from src.training.metrics import MetricsTracker
    tracker = MetricsTracker(device=device)
    
    logits = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    labels = torch.tensor([0, 1])
    
    tracker.update(logits, labels)
    metrics = tracker.compute()
    
    assert hasattr(metrics, 'pauc')
    assert metrics.pauc == 1.0
