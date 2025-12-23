import numpy as np
import pytest
import torch
from src.evaluation.metrics import calculate_eer, calculate_pauc

def test_calculate_eer_basic():
    # Perfect predictions
    labels = np.array([1, 1, 0, 0])
    logits = np.array([
        [-2.0, 2.0], # class 1
        [-2.0, 2.0], # class 1
        [2.0, -2.0], # class 0
        [2.0, -2.0], # class 0
    ])
    eer = calculate_eer(logits, labels)
    assert eer == 0.0

def test_calculate_eer_worst():
    # Worst predictions
    labels = np.array([1, 1, 0, 0])
    logits = np.array([
        [2.0, -2.0], # predicted 0, true 1
        [2.0, -2.0], # predicted 0, true 1
        [-2.0, 2.0], # predicted 1, true 0
        [-2.0, 2.0], # predicted 1, true 0
    ])
    eer = calculate_eer(logits, labels)
    assert eer == 1.0

def test_calculate_eer_random():
    labels = np.array([1, 0] * 50)
    logits = np.random.randn(100, 2)
    eer = calculate_eer(logits, labels)
    assert 0.0 <= eer <= 1.0

def test_calculate_pauc_basic():
    # Simple case: perfect predictions
    labels = np.array([1, 1, 0, 0])
    # Probs: [0.9, 0.8, 0.1, 0.2]
    logits = np.array([
        [-1.0, 1.0],  # p=0.88
        [-0.5, 0.5],  # p=0.73
        [1.0, -1.0],  # p=0.12
        [0.5, -0.5],  # p=0.27
    ])
    
    pauc = calculate_pauc(logits, labels, fpr_max=0.5)
    assert 0.0 <= pauc <= 1.0
    # With perfect or near perfect, pAUC should be high
    assert pauc > 0.5

def test_calculate_pauc_worst():
    # Worst case: opposite predictions
    labels = np.array([1, 0])
    logits = np.array([
        [1.0, -1.0], # Label 1, predicted 0
        [-1.0, 1.0], # Label 0, predicted 1
    ])
    
    pauc = calculate_pauc(logits, labels, fpr_max=0.5)
    assert pauc == 0.0

def test_calculate_pauc_fpr_max_boundary():
    # Test with very small fpr_max
    labels = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    logits = np.random.randn(10, 2)
    
    pauc = calculate_pauc(logits, labels, fpr_max=0.01)
    assert 0.0 <= pauc <= 1.0
