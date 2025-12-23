from typing import TYPE_CHECKING, Any, Dict, List, Sized, Tuple, cast

if TYPE_CHECKING:
    from src.evaluation.evaluator import ModelEvaluator
    from torch.utils.data import Dataset

import time
from pathlib import Path

import numpy as np
import structlog
import torch

from src.evaluation.types import EvaluationResult
from src.training.metrics import MetricResults

logger = structlog.get_logger(__name__)


def evaluate_dataset(
    evaluator: Any,  # type: ignore[arg-type]
    dataset: Any,  # type: ignore[arg-type]
    threshold: float = 0.5,
    batch_size: int = 32
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

    def collate_fn(
        batch: List[Tuple[torch.Tensor, int, Dict[str, Any]]]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
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
    # Calculate overall metrics
    all_preds: List[torch.Tensor] = []
    all_targs: List[torch.Tensor] = []
    all_logits = []
    results = []

    logger.info(f"Evaluating dataset with {len(cast(Sized, dataset))} samples...")

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
            all_preds.append(logits.cpu())
            all_targs.append(targets.cpu())
            all_logits.append(logits.cpu())

            # Create individual results
            probabilities = torch.softmax(logits, dim=1)
            confidences = probabilities[:, 1].cpu().numpy()
            predicted_classes = (confidences >= threshold).astype(int)

            for i, (confidence, pred_class, logit, meta, audio, target) in enumerate(
                zip(
                    confidences,
                    predicted_classes,
                    logits.cpu().numpy(),
                    metadata,
                    inputs.cpu().numpy(),
                    targets.cpu().numpy(),
                )
            ):
                results.append(
                    EvaluationResult(
                        filename=Path(meta["path"]).name if "path" in meta else f"sample_{batch_idx}_{i}",
                        prediction="Positive" if pred_class == 1 else "Negative",
                        confidence=float(confidence),
                        latency_ms=batch_latency,
                        logits=logit,
                        label=int(target),
                        raw_audio=audio.squeeze() if audio.ndim > 1 else audio,
                        full_path=str(meta["path"]) if "path" in meta else None,
                    )
                )

    # Calculate overall metrics - accumulate in lists first
    all_preds_raw = torch.stack([torch.tensor(r.logits) for r in results]) 
    all_targs_raw = torch.tensor([r.label for r in results])

    # Log label distribution for verification
    pos_count = (all_targs_raw == 1).sum().item()
    neg_count = (all_targs_raw == 0).sum().item()
    logger.info(f"Test Set Distribution: Positive (1)={pos_count}, Negative (0)={neg_count}")

    metrics = evaluator.metrics_calculator.calculate(all_preds_raw, all_targs_raw, threshold=threshold)

    logger.info(f"Evaluation complete: {metrics}")

    return metrics, results


def get_roc_curve_data(
    evaluator: "ModelEvaluator", dataset: "Dataset", batch_size: int = 32
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

    def collate_fn(
        batch: List[Tuple[torch.Tensor, int, Dict[str, Any]]]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
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

    all_confidences_arr = np.array(all_confidences)
    all_targets_arr = np.array(all_targets)

    # Calculate ROC curve
    thresholds = np.linspace(0, 1, 100)
    fpr_list = []
    tpr_list = []

    for threshold in thresholds:
        predictions = (all_confidences_arr >= threshold).astype(int)

        tp = ((predictions == 1) & (all_targets_arr == 1)).sum()
        tn = ((predictions == 0) & (all_targets_arr == 0)).sum()
        fp = ((predictions == 1) & (all_targets_arr == 0)).sum()
        fn = ((predictions == 0) & (all_targets_arr == 1)).sum()

        # True Positive Rate (Recall)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # False Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        fpr_list.append(fpr)
        tpr_list.append(tpr)

    return np.array(fpr_list), np.array(tpr_list), thresholds
