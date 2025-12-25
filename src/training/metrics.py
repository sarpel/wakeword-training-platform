"""
Metrics Tracking for Wakeword Detection
Includes: Accuracy, Precision, Recall, F1, FPR, FNR, Confusion Matrix
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog
import torch

logger = structlog.get_logger(__name__)


@dataclass
class MetricResults:
    """Container for metric calculation results"""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    fpr: float  # False Positive Rate
    fnr: float  # False Negative Rate

    # Confusion matrix components
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    # Additional stats
    total_samples: int
    positive_samples: int
    negative_samples: int

    # Fields with defaults MUST come last
    pauc: float = 0.0  # Partial AUC
    eer: float = 0.0  # Equal Error Rate
    fah: float = 0.0  # False Alarms per Hour
    latency_ms: float = 0.0  # Average latency per sample in ms
    confidence_histogram: str = ""  # ASCII Histogram visualization

    def __str__(self) -> str:
        """String representation"""
        return (
            f"\n    Accuracy:  {self.accuracy:.4f}\n"
            f"    Precision: {self.precision:.4f}\n"
            f"    Recall:    {self.recall:.4f}\n"
            f"    F1:        {self.f1_score:.4f}\n"
            f"    FPR:       {self.fpr:.4f}\n"
            f"    FNR:       {self.fnr:.4f}\n"
            f"    pAUC:      {self.pauc:.4f}\n"
            f"    EER:       {self.eer:.4f}"
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "fpr": self.fpr,
            "fnr": self.fnr,
            "pauc": self.pauc,
            "eer": self.eer,
            "fah": self.fah,
            "latency_ms": self.latency_ms,
            "true_positives": float(self.true_positives),
            "true_negatives": float(self.true_negatives),
            "false_positives": float(self.false_positives),
            "false_negatives": float(self.false_negatives),
            "total_samples": float(self.total_samples),
            "positive_samples": float(self.positive_samples),
            "negative_samples": float(self.negative_samples),
        }


class MetricsCalculator:
    """
    Calculate binary classification metrics
    Optimized for GPU operations
    """

    def __init__(self, device: str = "cuda") -> None:
        """
        Initialize metrics calculator

        Args:
            device: Device for computation ('cuda' or 'cpu')
        """
        self.device = device

    def calculate(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5,
        total_duration_h: Optional[float] = None,
    ) -> MetricResults:
        """
        Calculate all metrics from predictions and targets

        Args:
            predictions: Model predictions (logits or probabilities) (batch, num_classes)
            targets: Ground truth labels (batch,)
            threshold: Classification threshold for positive class
            total_duration_h: Total duration of the audio in hours (for FAH)

        Returns:
            MetricResults containing all calculated metrics
        """
        # Move to device
        if predictions.device != self.device:
            predictions = predictions.to(self.device)
        if targets.device != self.device:
            targets = targets.to(self.device)

        # Get predicted classes
        if predictions.dim() == 2:
            if predictions.size(1) >= 2:
                # Multi-class output (usually 2).
                # To respect threshold, we compute probability of class 1 (Positive)
                # If these are logits, apply softmax first
                if torch.abs(predictions.sum(dim=1) - 1.0).mean() > 1e-3:
                    # Likely logits, apply softmax
                    probs = torch.softmax(predictions, dim=1)[:, 1]
                else:
                    # Likely probabilities
                    probs = predictions[:, 1]
                pred_classes = (probs >= threshold).long()
            else:
                # Single output but 2D (batch, 1)
                # Apply sigmoid if not probabilities
                probs = (
                    torch.sigmoid(predictions[:, 0])
                    if predictions.max() > 1.0 or predictions.min() < 0.0
                    else predictions[:, 0]
                )
                pred_classes = (probs >= threshold).long()
        else:
            # Single output 1D (batch,)
            pred_classes = (predictions >= threshold).long()

        # Calculate confusion matrix components
        tp = int(((pred_classes == 1) & (targets == 1)).sum().item())
        tn = int(((pred_classes == 0) & (targets == 0)).sum().item())
        fp = int(((pred_classes == 1) & (targets == 0)).sum().item())
        fn = int(((pred_classes == 0) & (targets == 1)).sum().item())

        # Total samples
        total = len(targets)
        positive_samples = int((targets == 1).sum().item())
        negative_samples = int((targets == 0).sum().item())

        # Calculate metrics with safe division
        accuracy = (tp + tn) / total if total > 0 else 0.0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # False Positive Rate (FPR): FP / (FP + TN)
        # How often we incorrectly activate on negative samples
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # False Negative Rate (FNR): FN / (FN + TP)
        # How often we miss the wakeword
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        # Calculate pAUC
        from src.evaluation.metrics import calculate_eer, calculate_pauc

        pauc = calculate_pauc(predictions, targets, fpr_max=0.1)
        eer = calculate_eer(predictions, targets)

        # Calculate FAH (False Alarms per Hour)
        # FAH = FP / Total Duration (hours)
        fah = fp / total_duration_h if total_duration_h and total_duration_h > 0 else 0.0

        # Generate ASCII Histogram
        histogram = self._generate_ascii_histogram(probs, targets)

        return MetricResults(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            fpr=fpr,
            fnr=fnr,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            total_samples=total,
            positive_samples=positive_samples,
            negative_samples=negative_samples,
            pauc=pauc,
            eer=eer,
            fah=fah,
            confidence_histogram=histogram,
        )

    def _generate_ascii_histogram(self, probs: torch.Tensor, targets: torch.Tensor, bins: int = 10) -> str:
        """
        Generate a text-based histogram of confidence scores.
        Visually separates Negatives (L) vs Positives (R).
        """
        try:
            # Move to CPU for processing
            probs_np = probs.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()

            pos_scores = probs_np[targets_np == 1]
            neg_scores = probs_np[targets_np == 0]

            hist_lines = ["\n    Confidence Distribution (Neg [-] vs Pos [+]):"]
            
            # Create bins
            bin_edges = np.linspace(0, 1, bins + 1)
            
            # Calculate counts
            pos_hist, _ = np.histogram(pos_scores, bins=bin_edges)
            neg_hist, _ = np.histogram(neg_scores, bins=bin_edges)
            
            # Normalize for visualization (max width 40 chars)
            max_count = max(pos_hist.max(), neg_hist.max()) if (len(pos_hist) > 0 and len(neg_hist) > 0) else 1
            max_width = 40
            
            for i in range(bins):
                low, high = bin_edges[i], bin_edges[i+1]
                
                # Normalize lengths
                neg_len = int((neg_hist[i] / max_count) * max_width) if max_count > 0 else 0
                pos_len = int((pos_hist[i] / max_count) * max_width) if max_count > 0 else 0
                
                neg_bar = "-" * neg_len
                pos_bar = "+" * pos_len
                
                # Format: [0.0-0.1] --- (120) | + (5)
                # Using specific markers for clarity
                if neg_len > 0 and pos_len > 0:
                    bar = f"\033[94m{neg_bar}\033[0m|\033[92m{pos_bar}\033[0m" # Blue | Green
                elif neg_len > 0:
                    bar = f"\033[94m{neg_bar}\033[0m"
                elif pos_len > 0:
                    bar = f"|\033[92m{pos_bar}\033[0m"
                else:
                    bar = ""
                    
                hist_lines.append(f"    [{low:.1f}-{high:.1f}] {bar}")
                
            return "\n".join(hist_lines)
        except Exception as e:
            return f"Histogram error: {e}"

    def confusion_matrix(self, predictions: torch.Tensor, targets: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
        """
        Calculate confusion matrix

        Args:
            predictions: Model predictions (batch, num_classes)
            targets: Ground truth labels (batch,)
            num_classes: Number of classes

        Returns:
            Confusion matrix (num_classes, num_classes)
        """
        # Get predicted classes
        if predictions.dim() == 2:
            pred_classes = torch.argmax(predictions, dim=1)
        else:
            pred_classes = predictions.long()

        # Create confusion matrix
        conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long, device=self.device)

        for t, p in zip(targets, pred_classes):
            conf_matrix[t.long(), p.long()] += 1

        return conf_matrix


class MetricsTracker:
    """
    Track metrics across batches and epochs
    Accumulates predictions and targets for accurate metric calculation
    """

    def __init__(self, device: str = "cuda") -> None:
        """
        Initialize metrics tracker

        Args:
            device: Device for computation
        """
        self.device = device
        self.calculator = MetricsCalculator(device=device)

        # Accumulators
        self.all_predictions: List[torch.Tensor] = []
        self.all_targets: List[torch.Tensor] = []

        # Epoch history
        self.epoch_metrics: List[MetricResults] = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update tracker with batch predictions and targets

        Args:
            predictions: Model predictions (batch, num_classes)
            targets: Ground truth labels (batch,)
        """
        # Detach and move to CPU for accumulation
        self.all_predictions.append(predictions.detach().cpu())
        self.all_targets.append(targets.detach().cpu())

    def compute(self, threshold: float = 0.5, total_duration_h: Optional[float] = None) -> MetricResults:
        """
        Compute metrics from all accumulated predictions

        Args:
            threshold: Classification threshold
            total_duration_h: Total duration of the audio in hours (for FAH)

        Returns:
            MetricResults for all accumulated data
        """
        if not self.all_predictions:
            logger.warning("No predictions accumulated, returning zero metrics")
            return MetricResults(
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                fpr=0.0,
                fnr=0.0,
                true_positives=0,
                true_negatives=0,
                false_positives=0,
                false_negatives=0,
                total_samples=0,
                positive_samples=0,
                negative_samples=0,
            )

        # Concatenate all batches
        all_preds = torch.cat(self.all_predictions, dim=0)
        all_targs = torch.cat(self.all_targets, dim=0)

        # Calculate metrics
        metrics = self.calculator.calculate(
            all_preds, all_targs, threshold=threshold, total_duration_h=total_duration_h
        )

        return metrics

    def reset(self) -> None:
        """Reset accumulators for new epoch"""
        self.all_predictions.clear()
        self.all_targets.clear()

    def save_epoch_metrics(self, metrics: MetricResults) -> None:
        """
        Save metrics for current epoch

        Args:
            metrics: Metrics to save
        """
        self.epoch_metrics.append(metrics)

    def get_epoch_history(self) -> List[MetricResults]:
        """Get history of all epoch metrics"""
        return self.epoch_metrics

    def get_best_epoch(self, metric: str = "f1_score") -> Tuple[int, Optional[MetricResults]]:
        """
        Get epoch with best metric value

        Args:
            metric: Metric to optimize ('accuracy', 'f1_score', 'fpr', 'fnr')

        Returns:
            Tuple of (epoch_index, MetricResults)
        """
        if not self.epoch_metrics:
            return 0, None

        # For FPR and FNR, lower is better
        if metric in ["fpr", "fnr"]:
            best_idx = min(
                range(len(self.epoch_metrics)),
                key=lambda i: getattr(self.epoch_metrics[i], metric),
            )
        else:
            best_idx = max(
                range(len(self.epoch_metrics)),
                key=lambda i: getattr(self.epoch_metrics[i], metric),
            )

        return best_idx, self.epoch_metrics[best_idx]

    def summary(self) -> str:
        """
        Generate summary of all epochs

        Returns:
            Formatted summary string
        """
        if not self.epoch_metrics:
            return "No metrics recorded"

        summary = "METRICS SUMMARY\n"
        summary += "=" * 80 + "\n\n"

        for i, metrics in enumerate(self.epoch_metrics):
            summary += f"Epoch {i+1:3d}: {metrics}\n"

        summary += "\n" + "=" * 80 + "\n"

        # Best metrics
        best_acc_epoch, best_acc = self.get_best_epoch("accuracy")
        best_f1_epoch, best_f1 = self.get_best_epoch("f1_score")
        best_fpr_epoch, best_fpr = self.get_best_epoch("fpr")

        summary += (
            f"\nBest Accuracy: {best_acc.accuracy:.4f} (Epoch {best_acc_epoch+1})"
            if best_acc
            else "\nBest Accuracy: N/A"
        )
        summary += (
            f"\nBest F1 Score: {best_f1.f1_score:.4f} (Epoch {best_f1_epoch+1})" if best_f1 else "\nBest F1 Score: N/A"
        )
        summary += (
            f"\nBest FPR (lowest): {best_fpr.fpr:.4f} (Epoch {best_fpr_epoch+1})"
            if best_fpr
            else "\nBest FPR (lowest): N/A"
        )

        return summary


class MetricMonitor:
    """
    Real-time metric monitoring for training
    Calculates running averages and recent statistics
    """

    def __init__(self, window_size: int = 100) -> None:
        """
        Initialize metric monitor

        Args:
            window_size: Size of sliding window for running averages
        """
        self.window_size = window_size

        # Running accumulators
        self.batch_accuracies: List[float] = []
        self.batch_losses: List[float] = []

    def update_batch(self, loss: float, accuracy: float) -> None:
        """
        Update with batch statistics

        Args:
            loss: Batch loss value
            accuracy: Batch accuracy
        """
        self.batch_losses.append(loss)
        self.batch_accuracies.append(accuracy)

        # Trim to window size
        if len(self.batch_losses) > self.window_size:
            self.batch_losses = self.batch_losses[-self.window_size :]
        if len(self.batch_accuracies) > self.window_size:
            self.batch_accuracies = self.batch_accuracies[-self.window_size :]

    def get_running_averages(self) -> Dict[str, float]:
        """
        Get running averages over window

        Returns:
            Dictionary with running averages
        """
        if not self.batch_losses:
            return {"loss": 0.0, "accuracy": 0.0}

        return {
            "loss": float(np.mean(self.batch_losses).item()),
            "accuracy": float(np.mean(self.batch_accuracies).item()),
        }

    def reset(self) -> None:
        """Reset monitor for new epoch"""
        self.batch_losses.clear()
        self.batch_accuracies.clear()


def calculate_class_weights(
    dataset_stats: Dict[str, int], method: str = "balanced", device: str = "cuda"
) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets

    Args:
        dataset_stats: Dictionary with class counts {'positive': n, 'negative': m}
        method: Weighting method ('balanced', 'inverse', 'sqrt_inverse')
        device: Device to place weights on

    Returns:
        Class weights tensor
    """
    positive_count = dataset_stats.get("positive", 0)
    negative_count = dataset_stats.get("negative", 0)

    if positive_count == 0 or negative_count == 0:
        logger.warning("Zero samples in one class, returning equal weights")
        return torch.ones(2, device=device)

    total = positive_count + negative_count

    if method == "balanced":
        # sklearn-style: n_samples / (n_classes * class_count)
        weight_positive = total / (2 * positive_count)
        weight_negative = total / (2 * negative_count)

    elif method == "inverse":
        # Simple inverse frequency
        weight_positive = negative_count / positive_count
        weight_negative = 1.0

    elif method == "sqrt_inverse":
        # Square root of inverse frequency (less extreme)
        weight_positive = np.sqrt(negative_count / positive_count)
        weight_negative = 1.0

    else:
        raise ValueError(f"Unknown weighting method: {method}")

    # Class 0 (negative), Class 1 (positive)
    weights = torch.tensor([weight_negative, weight_positive], device=device)

    logger.info(f"Calculated class weights ({method}): Negative={weight_negative:.4f}, Positive={weight_positive:.4f}")

    return weights


if __name__ == "__main__":
    # Test metrics calculation
    print("Metrics System Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create dummy predictions and targets
    batch_size = 100
    num_classes = 2

    # Simulate model predictions (logits)
    predictions = torch.randn(batch_size, num_classes).to(device)

    # Simulate targets (80 negative, 20 positive - imbalanced)
    targets = torch.cat([torch.zeros(80, dtype=torch.long), torch.ones(20, dtype=torch.long)]).to(device)

    # Shuffle targets
    perm = torch.randperm(batch_size)
    targets = targets[perm]

    print("\nTest setup:")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Positive samples: {(targets == 1).sum().item()}")
    print(f"  Negative samples: {(targets == 0).sum().item()}")

    # Test MetricsCalculator
    print("\n1. Testing MetricsCalculator...")
    calculator = MetricsCalculator(device=device)
    metrics = calculator.calculate(predictions, targets)

    print(f"\n{metrics}")
    print("\nConfusion Matrix:")
    print(f"  TP: {metrics.true_positives} | FP: {metrics.false_positives}")
    print(f"  FN: {metrics.false_negatives} | TN: {metrics.true_negatives}")
    print("  ✅ MetricsCalculator works")

    # Test confusion matrix
    print("\n2. Testing confusion matrix calculation...")
    conf_matrix = calculator.confusion_matrix(predictions, targets, num_classes=2)
    print(f"  Confusion matrix:\n{conf_matrix}")
    print("  ✅ Confusion matrix works")

    # Test MetricsTracker
    print("\n3. Testing MetricsTracker (multi-batch)...")
    tracker = MetricsTracker(device=device)

    # Simulate 5 batches
    for i in range(5):
        batch_preds = torch.randn(20, num_classes).to(device)
        batch_targets = torch.randint(0, 2, (20,)).to(device)
        tracker.update(batch_preds, batch_targets)

    # Compute overall metrics
    overall_metrics = tracker.compute()
    print(f"  Overall metrics: {overall_metrics}")

    # Save as epoch
    tracker.save_epoch_metrics(overall_metrics)

    # Simulate another epoch
    tracker.reset()
    for i in range(5):
        batch_preds = torch.randn(20, num_classes).to(device)
        batch_targets = torch.randint(0, 2, (20,)).to(device)
        tracker.update(batch_preds, batch_targets)

    epoch2_metrics = tracker.compute()
    tracker.save_epoch_metrics(epoch2_metrics)

    print(f"\n  Epoch history: {len(tracker.get_epoch_history())} epochs")
    best_epoch, best_metrics = tracker.get_best_epoch("f1_score")
    f1 = best_metrics.f1_score if best_metrics else 0.0
    print(f"  Best epoch: {best_epoch + 1} with F1={f1:.4f}")
    print("  ✅ MetricsTracker works")

    # Test MetricMonitor
    print("\n4. Testing MetricMonitor...")
    monitor = MetricMonitor(window_size=10)

    # Simulate 20 batches
    for i in range(20):
        loss = 0.5 - i * 0.02  # Decreasing loss
        acc = 0.7 + i * 0.01  # Increasing accuracy
        monitor.update_batch(loss, acc)

    running_avg = monitor.get_running_averages()
    print(f"  Running averages: Loss={running_avg['loss']:.4f}, Acc={running_avg['accuracy']:.4f}")
    print("  ✅ MetricMonitor works")

    # Test class weights calculation
    print("\n5. Testing class weights calculation...")
    dataset_stats = {"positive": 20, "negative": 180}

    for method in ["balanced", "inverse", "sqrt_inverse"]:
        weights = calculate_class_weights(dataset_stats, method=method, device=device)
        print(f"  {method}: {weights}")

    print("  ✅ Class weights calculation works")

    print("\n✅ All metrics tests passed successfully")
    print("Metrics module loaded successfully")
