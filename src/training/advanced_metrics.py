"""
Advanced Metrics for Wakeword Detection
Includes: FAH, threshold selection, EER, pAUC, DET curves
"""
import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging
from sklearn.metrics import roc_curve, auc, roc_auc_score

logger = logging.getLogger(__name__)


@dataclass
class ThresholdMetrics:
    """Metrics at a specific threshold"""
    threshold: float
    tpr: float  # True Positive Rate (Recall)
    fpr: float  # False Positive Rate
    fnr: float  # False Negative Rate
    precision: float
    f1_score: float
    fah: float  # False Alarms per Hour
    accuracy: float


def calculate_fah(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
    total_seconds: float
) -> float:
    """
    Calculate False Alarms per Hour

    Args:
        logits: Model logits (N, 2) - REVERSED format [positive, negative]
        labels: Ground truth labels (N,)
        threshold: Classification threshold
        total_seconds: Total duration of audio in seconds

    Returns:
        False alarms per hour
    """
    # Get probabilities for positive class (index 0 due to reversed logits)
    probs = torch.softmax(logits, dim=1)[:, 0]

    # Predictions at threshold
    predictions = (probs >= threshold).long()

    # False positives
    fp = ((predictions == 1) & (labels == 0)).sum().item()

    # Calculate FAH
    total_hours = total_seconds / 3600.0
    fah = fp / (total_hours + 1e-9)

    return fah


def find_threshold_for_target_fah(
    logits: torch.Tensor,
    labels: torch.Tensor,
    total_seconds: float,
    target_fah: float,
    step: float = 0.0025
) -> Tuple[float, float]:
    """
    Find threshold that achieves target FAH while maximizing TPR

    Args:
        logits: Model logits (N, 2) - REVERSED format [positive, negative]
        labels: Ground truth labels (N,)
        total_seconds: Total audio duration
        target_fah: Target false alarms per hour
        step: Grid search step size

    Returns:
        Tuple of (threshold, tpr)
    """
    probs = torch.softmax(logits, dim=1)[:, 0]

    thresholds = torch.linspace(0, 1, int(1/step) + 1)
    best_threshold = 0.5
    best_tpr = 0.0

    total_hours = total_seconds / 3600.0
    positive_count = (labels == 1).sum().item()

    for threshold in thresholds:
        # Predictions
        predictions = (probs >= threshold).long()

        # Calculate FP and TP
        fp = ((predictions == 1) & (labels == 0)).sum().item()
        tp = ((predictions == 1) & (labels == 1)).sum().item()

        # Calculate FAH and TPR
        fah = fp / (total_hours + 1e-9)
        tpr = tp / max(positive_count, 1)

        # Check if FAH meets target
        if fah <= target_fah and tpr > best_tpr:
            best_threshold = threshold.item()
            best_tpr = tpr

    return best_threshold, best_tpr


def calculate_eer(
    logits: torch.Tensor,
    labels: torch.Tensor
) -> Tuple[float, float]:
    """
    Calculate Equal Error Rate (EER)

    The point where FPR = FNR

    Args:
        logits: Model logits - REVERSED format [positive, negative]
        labels: Ground truth labels

    Returns:
        Tuple of (eer, threshold)
    """
    probs = torch.softmax(logits, dim=1)[:, 0].cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Calculate FPR and TPR
    fpr, tpr, thresholds = roc_curve(labels_np, probs)

    # FNR = 1 - TPR
    fnr = 1 - tpr

    # Find where FPR = FNR
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]

    return float(eer), float(eer_threshold)


def calculate_pauc(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_fpr: float = 0.1
) -> float:
    """
    Calculate partial Area Under ROC Curve (pAUC)

    Focuses on low FPR region which is critical for wakeword detection

    Args:
        logits: Model logits - REVERSED format [positive, negative]
        labels: Ground truth labels
        max_fpr: Maximum FPR for partial AUC

    Returns:
        Partial AUC value
    """
    probs = torch.softmax(logits, dim=1)[:, 0].cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(labels_np, probs)

    # Find index where FPR exceeds max_fpr
    idx = np.where(fpr <= max_fpr)[0]

    if len(idx) == 0:
        return 0.0

    # Calculate partial AUC
    fpr_partial = fpr[idx]
    tpr_partial = tpr[idx]

    # Normalize to [0, 1] range
    pauc = auc(fpr_partial, tpr_partial) / max_fpr

    return float(pauc)


def calculate_metrics_at_threshold(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
    total_seconds: Optional[float] = None
) -> ThresholdMetrics:
    """
    Calculate all metrics at a specific threshold

    Args:
        logits: Model logits - REVERSED format [positive, negative]
        labels: Ground truth labels
        threshold: Classification threshold
        total_seconds: Total audio duration (for FAH calculation)

    Returns:
        ThresholdMetrics object
    """
    probs = torch.softmax(logits, dim=1)[:, 0]
    predictions = (probs >= threshold).long()

    # Confusion matrix
    tp = ((predictions == 1) & (labels == 1)).sum().item()
    tn = ((predictions == 0) & (labels == 0)).sum().item()
    fp = ((predictions == 1) & (labels == 0)).sum().item()
    fn = ((predictions == 0) & (labels == 1)).sum().item()

    # Metrics
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)

    precision = tp / max(tp + fp, 1)
    f1 = 2 * (precision * tpr) / max(precision + tpr, 1e-9)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    # Calculate FAH if total_seconds provided
    if total_seconds is not None:
        total_hours = total_seconds / 3600.0
        fah = fp / max(total_hours, 1e-9)
    else:
        fah = 0.0

    return ThresholdMetrics(
        threshold=threshold,
        tpr=tpr,
        fpr=fpr,
        fnr=fnr,
        precision=precision,
        f1_score=f1,
        fah=fah,
        accuracy=accuracy
    )


def calculate_det_curve(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Detection Error Tradeoff (DET) curve

    DET curve plots FNR vs FPR (both on normal deviate scale in practice)

    Args:
        logits: Model logits - REVERSED format [positive, negative]
        labels: Ground truth labels
        num_points: Number of points in curve

    Returns:
        Tuple of (fpr_points, fnr_points)
    """
    probs = torch.softmax(logits, dim=1)[:, 0].cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(labels_np, probs)

    # FNR = 1 - TPR
    fnr = 1 - tpr

    return fpr, fnr


def grid_search_threshold(
    logits: torch.Tensor,
    labels: torch.Tensor,
    total_seconds: float,
    step: float = 0.0025
) -> List[ThresholdMetrics]:
    """
    Perform grid search over thresholds

    Args:
        logits: Model logits
        labels: Ground truth labels
        total_seconds: Total audio duration
        step: Grid step size

    Returns:
        List of ThresholdMetrics for each threshold
    """
    thresholds = torch.linspace(0, 1, int(1/step) + 1)
    metrics_list = []

    for threshold in thresholds:
        metrics = calculate_metrics_at_threshold(
            logits, labels, threshold.item(), total_seconds
        )
        metrics_list.append(metrics)

    return metrics_list


def find_operating_point(
    logits: torch.Tensor,
    labels: torch.Tensor,
    total_seconds: float,
    target_fah: float,
    step: float = 0.0025
) -> ThresholdMetrics:
    """
    Find operating point that meets target FAH while maximizing recall

    Args:
        logits: Model logits
        labels: Ground truth labels
        total_seconds: Total audio duration
        target_fah: Target false alarms per hour
        step: Grid step size

    Returns:
        ThresholdMetrics at operating point
    """
    # Grid search
    all_metrics = grid_search_threshold(logits, labels, total_seconds, step)

    # Find best threshold that meets FAH constraint
    best_metrics = None
    best_tpr = 0.0

    for metrics in all_metrics:
        if metrics.fah <= target_fah and metrics.tpr > best_tpr:
            best_metrics = metrics
            best_tpr = metrics.tpr

    if best_metrics is None:
        logger.warning(
            f"Could not find threshold meeting target FAH={target_fah}. "
            f"Returning threshold with lowest FAH."
        )
        best_metrics = min(all_metrics, key=lambda m: m.fah)

    return best_metrics


def calculate_comprehensive_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    total_seconds: float,
    target_fah: float = 1.0
) -> Dict:
    """
    Calculate comprehensive metrics suite

    Args:
        logits: Model logits - REVERSED format [positive, negative]
        labels: Ground truth labels
        total_seconds: Total audio duration
        target_fah: Target FAH for operating point

    Returns:
        Dictionary with all metrics
    """
    logger.info("Calculating comprehensive metrics...")

    # Standard ROC-AUC (use index 0 for positive class due to reversed logits)
    try:
        roc_auc = roc_auc_score(
            labels.cpu().numpy(),
            torch.softmax(logits, dim=1)[:, 0].cpu().numpy()
        )
    except:
        roc_auc = 0.0

    # EER
    eer, eer_threshold = calculate_eer(logits, labels)

    # pAUC (FPR <= 0.1)
    pauc = calculate_pauc(logits, labels, max_fpr=0.1)

    # Operating point at target FAH
    operating_point = find_operating_point(
        logits, labels, total_seconds, target_fah
    )

    # Metrics at EER threshold
    eer_metrics = calculate_metrics_at_threshold(
        logits, labels, eer_threshold, total_seconds
    )

    results = {
        'roc_auc': roc_auc,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'pauc_at_fpr_0.1': pauc,
        'operating_point': {
            'threshold': operating_point.threshold,
            'tpr': operating_point.tpr,
            'fpr': operating_point.fpr,
            'fnr': operating_point.fnr,
            'precision': operating_point.precision,
            'f1_score': operating_point.f1_score,
            'fah': operating_point.fah,
            'target_fah': target_fah
        },
        'eer_point': {
            'threshold': eer_threshold,
            'tpr': eer_metrics.tpr,
            'fpr': eer_metrics.fpr,
            'fnr': eer_metrics.fnr,
            'fah': eer_metrics.fah
        }
    }

    logger.info(f"Comprehensive metrics calculated:")
    logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  EER: {eer:.4f} @ threshold={eer_threshold:.4f}")
    logger.info(f"  pAUC (FPR<=0.1): {pauc:.4f}")
    logger.info(f"  Operating Point (FAH<={target_fah}):")
    logger.info(f"    Threshold: {operating_point.threshold:.4f}")
    logger.info(f"    TPR: {operating_point.tpr:.4f}")
    logger.info(f"    FPR: {operating_point.fpr:.4f}")
    logger.info(f"    FAH: {operating_point.fah:.2f}")

    return results


if __name__ == "__main__":
    # Test advanced metrics
    print("Advanced Metrics Test")
    print("=" * 60)

    # Create dummy data
    num_samples = 1000
    num_positive = 200

    # Simulate logits (random but biased toward correct class)
    logits = torch.randn(num_samples, 2)
    labels = torch.cat([
        torch.ones(num_positive),
        torch.zeros(num_samples - num_positive)
    ]).long()

    # Shuffle
    perm = torch.randperm(num_samples)
    logits = logits[perm]
    labels = labels[perm]

    # Bias logits toward correct answers
    for i in range(num_samples):
        logits[i, labels[i]] += 2.0

    print(f"Test data created:")
    print(f"  Samples: {num_samples}")
    print(f"  Positive: {num_positive}")
    print(f"  Negative: {num_samples - num_positive}")

    # Assume 10 hours of audio
    total_seconds = 10 * 3600

    # Test FAH calculation
    print(f"\n1. Testing FAH calculation...")
    fah = calculate_fah(logits, labels, threshold=0.5, total_seconds=total_seconds)
    print(f"  FAH @ threshold=0.5: {fah:.2f}")

    # Test threshold finding for target FAH
    print(f"\n2. Finding threshold for target FAH=1.0...")
    threshold, tpr = find_threshold_for_target_fah(
        logits, labels, total_seconds, target_fah=1.0
    )
    print(f"  Threshold: {threshold:.4f}")
    print(f"  TPR: {tpr:.4f}")

    # Test EER
    print(f"\n3. Calculating EER...")
    eer, eer_threshold = calculate_eer(logits, labels)
    print(f"  EER: {eer:.4f}")
    print(f"  EER threshold: {eer_threshold:.4f}")

    # Test pAUC
    print(f"\n4. Calculating pAUC...")
    pauc = calculate_pauc(logits, labels, max_fpr=0.1)
    print(f"  pAUC (FPR<=0.1): {pauc:.4f}")

    # Test operating point finding
    print(f"\n5. Finding operating point...")
    op_metrics = find_operating_point(
        logits, labels, total_seconds, target_fah=1.0
    )
    print(f"  Threshold: {op_metrics.threshold:.4f}")
    print(f"  TPR: {op_metrics.tpr:.4f}")
    print(f"  FPR: {op_metrics.fpr:.4f}")
    print(f"  FAH: {op_metrics.fah:.2f}")
    print(f"  F1: {op_metrics.f1_score:.4f}")

    # Test comprehensive metrics
    print(f"\n6. Calculating comprehensive metrics...")
    comp_metrics = calculate_comprehensive_metrics(
        logits, labels, total_seconds, target_fah=1.0
    )

    print("\n✅ Advanced metrics test complete")
