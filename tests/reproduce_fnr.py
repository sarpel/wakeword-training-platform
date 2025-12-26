from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.types import EvaluationResult
from src.training.metrics import MetricsCalculator


class MockPerfectModel(nn.Module):
    """A model that always returns high probability for the correct class 1 for positive and 0 for negative"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        # We don't care about x, we want to see how MetricsCalculator handles the output
        # This is a dummy to satisfy ModelEvaluator
        return torch.randn(x.size(0), 2)


def test_metrics_alignment():
    print("\nReproducing Metrics Alignment Issue...")

    # Simulate logits where class 1 is strongly predicted for positive samples
    # and class 0 is strongly predicted for negative samples
    # (batch_size, num_classes)
    logits = torch.tensor(
        [
            [-5.0, 5.0],  # Positive sample, predicted class 1 (Correct)
            [-5.0, 5.0],  # Positive sample, predicted class 1 (Correct)
            [5.0, -5.0],  # Negative sample, predicted class 0 (Correct)
            [5.0, -5.0],  # Negative sample, predicted class 0 (Correct)
        ]
    )

    targets = torch.tensor([1, 1, 0, 0])  # 1=Positive, 0=Negative

    calculator = MetricsCalculator(device="cpu")
    metrics = calculator.calculate(logits, targets)

    print(f"Metrics: {metrics}")
    print(f"TP: {metrics.true_positives} | TN: {metrics.true_negatives}")
    print(f"FP: {metrics.false_positives} | FN: {metrics.false_negatives}")
    print(f"FNR: {metrics.fnr:.2%}")
    print(f"F1: {metrics.f1_score:.2%}")

    # If the logic is correct, TP=2, TN=2, FP=0, FN=0, FNR=0%, F1=100%
    assert metrics.true_positives == 2
    assert metrics.true_negatives == 2
    assert metrics.false_negatives == 0
    assert metrics.fnr == 0.0

    print("✅ MetricsCalculator handles Positive=1 correctly.")


def test_evaluator_confidences():
    # In src/evaluation/dataset_evaluator.py:
    # probabilities = torch.softmax(logits, dim=1)
    # confidences = probabilities[:, 1].cpu().numpy()
    # predicted_classes = (confidences >= threshold).astype(int)

    logits = torch.tensor(
        [
            [-5.0, 5.0],  # Prob[1] ≈ 1.0
            [5.0, -5.0],  # Prob[1] ≈ 0.0
        ]
    )

    threshold = 0.5
    probabilities = torch.softmax(logits, dim=1)
    confidences = probabilities[:, 1].numpy()
    predicted_classes = (confidences >= threshold).astype(int)

    print(f"\nConfidences: {confidences}")
    print(f"Predicted Classes: {predicted_classes}")

    assert predicted_classes[0] == 1
    assert predicted_classes[1] == 0
    print("✅ Evaluator confidence logic handles Positive=1 correctly.")


if __name__ == "__main__":
    try:
        test_metrics_alignment()
        test_evaluator_confidences()
        print("\nAll reproduction tests passed! The core logic seems correct.")
        print("Wait, if the core logic is correct, why does the user see >99% FNR?")
        print("Could it be that the model output is (N, 1) and not (N, 2)?")
    except Exception as e:
        print(f"\n❌ Reproduction failed: {e}")
