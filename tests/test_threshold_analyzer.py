import pytest
import torch
import numpy as np
from src.evaluation.advanced_evaluator import ThresholdAnalyzer

def test_threshold_analyzer_initialization():
    """Test initializing ThresholdAnalyzer with valid inputs."""
    logits = torch.randn(100, 2)
    labels = torch.randint(0, 2, (100,))
    analyzer = ThresholdAnalyzer(logits, labels)
    assert analyzer.logits.shape == (100, 2)
    assert analyzer.labels.shape == (100,)

def test_compute_metrics_at_threshold():
    """Test computing metrics for a specific threshold."""
    # Create deterministic data
    # Class 1 is positive.
    # High logit for class 1 = high confidence
    logits = torch.tensor([
        [0.1, 0.9], # Conf ~0.7, True Pos
        [0.8, 0.2], # Conf ~0.35, True Neg
        [0.4, 0.6], # Conf ~0.55, False Pos (if label=0)
        [0.7, 0.3]  # Conf ~0.4, False Neg (if label=1)
    ])
    labels = torch.tensor([1, 0, 0, 1])
    
    analyzer = ThresholdAnalyzer(logits, labels)
    
    # Threshold 0.5
    # Sample 1: 0.7 > 0.5 -> Pos (TP)
    # Sample 2: 0.35 < 0.5 -> Neg (TN)
    # Sample 3: 0.55 > 0.5 -> Pos (FP)
    # Sample 4: 0.4 < 0.5 -> Neg (FN)
    # TP=1, TN=1, FP=1, FN=1
    # Precision = TP / (TP+FP) = 1/2 = 0.5
    # Recall = TP / (TP+FN) = 1/2 = 0.5
    
    metrics = analyzer.compute_at_threshold(0.5)
    assert metrics['precision'] == 0.5
    assert metrics['recall'] == 0.5
    assert metrics['tp'] == 1
    assert metrics['fp'] == 1

def test_analyze_threshold_range():
    """Test analyzing a range of thresholds."""
    logits = torch.randn(50, 2)
    labels = torch.randint(0, 2, (50,))
    analyzer = ThresholdAnalyzer(logits, labels)
    
    thresholds = np.linspace(0, 1, 11)
    results = analyzer.analyze_range(thresholds)
    
    assert len(results) == 11
    assert 'threshold' in results[0]
    assert 'precision' in results[0]
    assert 'recall' in results[0]
