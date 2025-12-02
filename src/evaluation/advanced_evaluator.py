from typing import TYPE_CHECKING, Any, Dict, List, Sized, Tuple, cast

if TYPE_CHECKING:
    from src.evaluation.evaluator import ModelEvaluator
    from torch.utils.data import Dataset

import structlog
import torch

from src.training.advanced_metrics import calculate_comprehensive_metrics

logger = structlog.get_logger(__name__)


def evaluate_with_advanced_metrics(
    evaluator: "ModelEvaluator",
    dataset: "Dataset",
    total_seconds: float,
    target_fah: float = 1.0,
    batch_size: int = 32,
) -> Dict:
    """
    Evaluate dataset with comprehensive production metrics

    Args:
        dataset: PyTorch Dataset with ground truth
        total_seconds: Total audio duration in dataset (for FAH calculation)
        target_fah: Target false alarms per hour for operating point
        batch_size: Batch size for processing

    Returns:
        Dictionary with comprehensive metrics including FAH, EER, pAUC
    """
    from torch.utils.data import DataLoader

    def collate_fn(
        batch: List[Tuple[torch.Tensor, int, Dict[str, Any]]]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """Custom collate function to handle metadata"""
        features, labels, metadata_list = zip(*batch)
        features = torch.stack(features)
        labels = torch.tensor(labels)
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

    all_logits = []
    all_targets = []

    logger.info(f"Running comprehensive evaluation on {len(cast(Sized, dataset))} samples...")

    # Collect all predictions
    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(evaluator.device)

            # NEW: GPU Processing Pipeline for Raw Audio
            # If input is raw audio (B, S) or (B, 1, S), run through AudioProcessor
            if inputs.ndim <= 3:
                inputs = evaluator.audio_processor(inputs)

            # Apply memory format optimization
            inputs = inputs.to(memory_format=torch.channels_last)

            with torch.cuda.amp.autocast():
                logits = evaluator.model(inputs)
            # Convert to float32 immediately after inference to ensure compatibility
            logits = logits.float()

            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    all_logits_tensor = torch.cat(all_logits, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)

    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(
        logits=all_logits_tensor,
        labels=all_targets_tensor,
        total_seconds=total_seconds,
        target_fah=target_fah,
    )

    logger.info("Advanced evaluation complete:")
    logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"  EER: {metrics['eer']:.4f}")
    logger.info(f"  pAUC (FPR≤0.1): {metrics['pauc_at_fpr_0.1']:.4f}")
    logger.info(f"  Operating Point:")
    logger.info(f"    Threshold: {metrics['operating_point']['threshold']:.4f}")
    logger.info(f"    TPR: {metrics['operating_point']['tpr']:.4f}")
    logger.info(f"    FPR: {metrics['operating_point']['fpr']:.4f}")
    logger.info(f"    FAH: {metrics['operating_point']['fah']:.2f}")

    return metrics
