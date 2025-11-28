from typing import List, Tuple
import time
from pathlib import Path

import numpy as np
import structlog
import torch

from src.evaluation.types import EvaluationResult
from src.training.metrics import MetricResults

logger = structlog.get_logger(__name__)


def evaluate_dataset(
    evaluator, dataset, threshold: float = 0.5, batch_size: int = 32
) -> Tuple[MetricResults, List[EvaluationResult]]:
    """
    Evaluate entire dataset with ground truth labels

    Args:
        dataset: PyTorch Dataset with ground truth
        threshold: Classification threshold
        batch_size: Batch size for processing

    Returns:
        Tuple of (MetricResults, List[EvaluationResult])
    """
    from torch.utils.data import DataLoader

    def collate_fn(batch):
        """Custom collate function to handle metadata"""
        features, labels, metadata_list = zip(*batch)

        # Stack features and labels
        features = torch.stack(features)
        labels = torch.tensor(labels)

        # Keep metadata as list of dicts
        return features, labels, list(metadata_list)

    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Single process for evaluation
        pin_memory=True,
        collate_fn=collate_fn,
    )

    all_predictions = []
    all_targets = []
    all_logits = []
    results = []

    logger.info(f"Evaluating dataset with {len(dataset)} samples...")

    # Evaluate
    with torch.no_grad():
        for batch_idx, (inputs, targets, metadata) in enumerate(loader):
            inputs = inputs.to(evaluator.device)
            targets = targets.to(evaluator.device)

            # NEW: GPU Processing Pipeline for Raw Audio
            # If input is raw audio (B, S) or (B, 1, S), run through AudioProcessor
            if inputs.ndim <= 3:
                inputs = evaluator.audio_processor(inputs)
            
            # Apply memory format optimization
            inputs = inputs.to(memory_format=torch.channels_last)

            # Inference
            start_time = time.time()

            with torch.cuda.amp.autocast():
                logits = evaluator.model(inputs)
            # Convert to float32 immediately after inference to ensure compatibility
            logits = logits.float()

            batch_latency = (time.time() - start_time) * 1000 / len(inputs)

            # Collect for metrics
            all_predictions.append(logits.cpu())
            all_targets.append(targets.cpu())
            all_logits.append(logits.cpu())

            # Create individual results
            probabilities = torch.softmax(logits, dim=1)
            confidences = probabilities[:, 1].cpu().numpy()
            predicted_classes = (confidences >= threshold).astype(int)

            for i, (confidence, pred_class, logit, meta) in enumerate(
                zip(confidences, predicted_classes, logits.cpu().numpy(), metadata)
            ):
                results.append(
                    EvaluationResult(
                        filename=Path(meta["path"]).name
                        if "path" in meta
                        else f"sample_{batch_idx}_{i}",
                        prediction="Positive" if pred_class == 1 else "Negative",
                        confidence=float(confidence),
                        latency_ms=batch_latency,
                        logits=logit,
                    )
                )

    # Calculate overall metrics
    all_preds = torch.cat(all_predictions, dim=0)
    all_targs = torch.cat(all_targets, dim=0)

    metrics = evaluator.metrics_calculator.calculate(
        all_preds, all_targs, threshold=threshold
    )

    logger.info(f"Evaluation complete: {metrics}")

    return metrics, results


def get_roc_curve_data(
    evaluator, dataset, batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate ROC curve data

    Args:
        dataset: PyTorch Dataset with ground truth
        batch_size: Batch size for processing

    Returns:
        Tuple of (fpr_array, tpr_array, thresholds)
    """
    from torch.utils.data import DataLoader

    def collate_fn(batch):
        """Custom collate function to handle metadata"""
        features, labels, metadata_list = zip(*batch)

        # Stack features and labels
        features = torch.stack(features)
        labels = torch.tensor(labels)

        # Keep metadata as list of dicts
        return features, labels, list(metadata_list)

    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    all_confidences = []
    all_targets = []

    # Collect predictions
    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(evaluator.device)

            # NEW: GPU Processing Pipeline for Raw Audio
            if inputs.ndim <= 3:
                inputs = evaluator.audio_processor(inputs)
            
            # Apply memory format optimization
            inputs = inputs.to(memory_format=torch.channels_last)

            with torch.cuda.amp.autocast():
                logits = evaluator.model(inputs)
            # Convert to float32 immediately after inference to ensure compatibility
            logits = logits.float()

            # Calculate probabilities
            probabilities = torch.softmax(logits, dim=1)
            confidences = probabilities[:, 1].cpu().numpy()

            all_confidences.extend(confidences)
            all_targets.extend(targets.cpu().numpy())

    all_confidences = np.array(all_confidences)
    all_targets = np.array(all_targets)

    # Calculate ROC curve
    thresholds = np.linspace(0, 1, 100)
    fpr_list = []
    tpr_list = []

    for threshold in thresholds:
        predictions = (all_confidences >= threshold).astype(int)

        tp = ((predictions == 1) & (all_targets == 1)).sum()
        tn = ((predictions == 0) & (all_targets == 0)).sum()
        fp = ((predictions == 1) & (all_targets == 0)).sum()
        fn = ((predictions == 0) & (all_targets == 1)).sum()

        # True Positive Rate (Recall)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # False Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        fpr_list.append(fpr)
        tpr_list.append(tpr)

    return np.array(fpr_list), np.array(tpr_list), thresholds