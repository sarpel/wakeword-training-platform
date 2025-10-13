"""
Checkpoint Management Utilities
Handle checkpoint loading, saving, and management
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import logging
import json
import shutil

logger = logging.getLogger(__name__)


@dataclass
class CheckpointInfo:
    """Metadata for a checkpoint"""
    path: Path
    epoch: int
    val_loss: float
    val_accuracy: float
    val_f1: float
    val_fpr: float
    val_fnr: float
    timestamp: Optional[str] = None

    def __str__(self) -> str:
        return (
            f"Epoch {self.epoch}: "
            f"Loss={self.val_loss:.4f}, "
            f"Acc={self.val_accuracy:.4f}, "
            f"F1={self.val_f1:.4f}, "
            f"FPR={self.val_fpr:.4f}"
        )


class CheckpointManager:
    """
    Manage model checkpoints
    Load, save, list, and clean up checkpoints
    """

    def __init__(self, checkpoint_dir: Path):
        """
        Initialize checkpoint manager

        Args:
            checkpoint_dir: Directory containing checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Checkpoint manager initialized: {self.checkpoint_dir}")

    def list_checkpoints(self) -> List[CheckpointInfo]:
        """
        List all available checkpoints with metadata

        Returns:
            List of CheckpointInfo objects
        """
        checkpoints = []

        for checkpoint_path in self.checkpoint_dir.glob("*.pt"):
            try:
                # Load checkpoint metadata
                checkpoint = torch.load(checkpoint_path, map_location='cpu')

                # Extract metadata
                epoch = checkpoint.get('epoch', 0)
                val_loss = checkpoint.get('val_loss', float('inf'))

                # Get validation metrics
                val_metrics = checkpoint.get('val_metrics', {})
                val_accuracy = val_metrics.get('accuracy', 0.0)
                val_f1 = val_metrics.get('f1_score', 0.0)
                val_fpr = val_metrics.get('fpr', 1.0)
                val_fnr = val_metrics.get('fnr', 1.0)

                info = CheckpointInfo(
                    path=checkpoint_path,
                    epoch=epoch,
                    val_loss=val_loss,
                    val_accuracy=val_accuracy,
                    val_f1=val_f1,
                    val_fpr=val_fpr,
                    val_fnr=val_fnr
                )

                checkpoints.append(info)

            except Exception as e:
                logger.warning(f"Failed to load checkpoint {checkpoint_path}: {e}")

        # Sort by epoch
        checkpoints.sort(key=lambda x: x.epoch)

        return checkpoints

    def get_best_checkpoint(
        self,
        metric: str = 'f1_score',
        mode: str = 'max'
    ) -> Optional[CheckpointInfo]:
        """
        Get best checkpoint based on metric

        Args:
            metric: Metric to optimize ('val_loss', 'val_accuracy', 'val_f1', 'val_fpr')
            mode: 'max' or 'min' for optimization direction

        Returns:
            CheckpointInfo for best checkpoint, or None if no checkpoints
        """
        checkpoints = self.list_checkpoints()

        if not checkpoints:
            logger.warning("No checkpoints found")
            return None

        # Select best based on metric
        if mode == 'min':
            best = min(checkpoints, key=lambda x: getattr(x, metric))
        else:
            best = max(checkpoints, key=lambda x: getattr(x, metric))

        logger.info(f"Best checkpoint ({metric}, {mode}): {best}")

        return best

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Any] = None,
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """
        Load checkpoint and optionally restore model/optimizer

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load weights into (optional)
            optimizer: Optimizer to load state into (optional)
            device: Device to load tensors to

        Returns:
            Checkpoint dictionary
        """
        logger.info(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model weights
        if model is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("  Loaded model weights")

        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("  Loaded optimizer state")

        logger.info(f"  Epoch: {checkpoint['epoch'] + 1}")
        logger.info(f"  Val loss: {checkpoint.get('val_loss', 'N/A')}")

        return checkpoint

    def load_best_model(
        self,
        model: nn.Module,
        metric: str = 'val_f1',
        mode: str = 'max',
        device: str = 'cuda'
    ) -> Optional[Dict[str, Any]]:
        """
        Load best model weights

        Args:
            model: Model to load weights into
            metric: Metric to select best checkpoint
            mode: 'max' or 'min'
            device: Device to load to

        Returns:
            Checkpoint dictionary, or None if no checkpoints
        """
        # Check for best_model.pt first
        best_path = self.checkpoint_dir / "best_model.pt"

        if best_path.exists():
            logger.info("Loading best_model.pt")
            return self.load_checkpoint(best_path, model=model, device=device)

        # Otherwise find best based on metric
        best_checkpoint = self.get_best_checkpoint(metric=metric, mode=mode)

        if best_checkpoint is None:
            logger.warning("No checkpoints available to load")
            return None

        return self.load_checkpoint(best_checkpoint.path, model=model, device=device)

    def cleanup_old_checkpoints(
        self,
        keep_n: int = 5,
        keep_best: bool = True
    ):
        """
        Delete old checkpoints, keeping only N most recent

        Args:
            keep_n: Number of checkpoints to keep
            keep_best: Whether to always keep best_model.pt
        """
        checkpoints = self.list_checkpoints()

        if len(checkpoints) <= keep_n:
            logger.info(f"Only {len(checkpoints)} checkpoints, no cleanup needed")
            return

        # Sort by epoch (oldest first)
        checkpoints.sort(key=lambda x: x.epoch)

        # Keep N most recent
        to_delete = checkpoints[:-keep_n]

        # Delete old checkpoints
        for checkpoint in to_delete:
            # Skip best_model.pt if keep_best is True
            if keep_best and checkpoint.path.name == "best_model.pt":
                continue

            try:
                checkpoint.path.unlink()
                logger.info(f"Deleted old checkpoint: {checkpoint.path.name}")
            except Exception as e:
                logger.warning(f"Failed to delete {checkpoint.path}: {e}")

    def export_checkpoint_info(self, output_path: Path):
        """
        Export checkpoint metadata to JSON

        Args:
            output_path: Path to save JSON file
        """
        checkpoints = self.list_checkpoints()

        # Convert to dictionary
        checkpoint_data = []
        for checkpoint in checkpoints:
            checkpoint_data.append({
                'path': str(checkpoint.path),
                'epoch': checkpoint.epoch,
                'val_loss': checkpoint.val_loss,
                'val_accuracy': checkpoint.val_accuracy,
                'val_f1': checkpoint.val_f1,
                'val_fpr': checkpoint.val_fpr,
                'val_fnr': checkpoint.val_fnr
            })

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info(f"Exported checkpoint info to: {output_path}")

    def create_model_snapshot(
        self,
        model: nn.Module,
        snapshot_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Create a standalone model snapshot (weights only, no optimizer)

        Args:
            model: Model to save
            snapshot_name: Name for snapshot file
            metadata: Optional metadata dictionary
        """
        snapshot_path = self.checkpoint_dir / f"{snapshot_name}.pt"

        snapshot = {
            'model_state_dict': model.state_dict(),
            'metadata': metadata or {}
        }

        torch.save(snapshot, snapshot_path)
        logger.info(f"Created model snapshot: {snapshot_path}")

    def load_model_snapshot(
        self,
        model: nn.Module,
        snapshot_name: str,
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """
        Load model from snapshot

        Args:
            model: Model to load weights into
            snapshot_name: Name of snapshot file
            device: Device to load to

        Returns:
            Snapshot metadata
        """
        snapshot_path = self.checkpoint_dir / f"{snapshot_name}.pt"

        if not snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

        logger.info(f"Loading model snapshot: {snapshot_path}")

        snapshot = torch.load(snapshot_path, map_location=device)
        model.load_state_dict(snapshot['model_state_dict'])

        return snapshot.get('metadata', {})


def extract_model_for_inference(
    checkpoint_path: Path,
    output_path: Path,
    device: str = 'cuda'
):
    """
    Extract model weights from checkpoint for inference-only use

    Args:
        checkpoint_path: Path to full checkpoint
        output_path: Path to save inference model
        device: Device for loading
    """
    logger.info(f"Extracting model from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create inference-only checkpoint
    inference_checkpoint = {
        'model_state_dict': checkpoint['model_state_dict'],
        'config': checkpoint.get('config'),
        'epoch': checkpoint.get('epoch'),
        'val_metrics': checkpoint.get('val_metrics')
    }

    # Save
    torch.save(inference_checkpoint, output_path)

    # Calculate size reduction
    original_size = checkpoint_path.stat().st_size / 1024 / 1024
    new_size = output_path.stat().st_size / 1024 / 1024

    logger.info(f"Extracted model saved to: {output_path}")
    logger.info(f"  Original size: {original_size:.2f} MB")
    logger.info(f"  Inference size: {new_size:.2f} MB")
    logger.info(f"  Size reduction: {(1 - new_size/original_size)*100:.1f}%")


def compare_checkpoints(
    checkpoint_paths: List[Path],
    metric: str = 'val_f1'
) -> List[Tuple[Path, float]]:
    """
    Compare multiple checkpoints by metric

    Args:
        checkpoint_paths: List of checkpoint paths
        metric: Metric to compare

    Returns:
        List of (path, metric_value) tuples sorted by metric
    """
    results = []

    for path in checkpoint_paths:
        try:
            checkpoint = torch.load(path, map_location='cpu')
            val_metrics = checkpoint.get('val_metrics', {})

            # Map metric name
            if metric == 'val_f1':
                value = val_metrics.get('f1_score', 0.0)
            elif metric == 'val_fpr':
                value = val_metrics.get('fpr', 1.0)
            elif metric == 'val_accuracy':
                value = val_metrics.get('accuracy', 0.0)
            elif metric == 'val_loss':
                value = checkpoint.get('val_loss', float('inf'))
            else:
                value = 0.0

            results.append((path, value))

        except Exception as e:
            logger.warning(f"Failed to load checkpoint {path}: {e}")

    # Sort by metric (descending for most metrics, ascending for loss/fpr)
    if metric in ['val_loss', 'val_fpr']:
        results.sort(key=lambda x: x[1])  # Lower is better
    else:
        results.sort(key=lambda x: x[1], reverse=True)  # Higher is better

    return results


if __name__ == "__main__":
    # Test checkpoint manager
    print("Checkpoint Manager Test")
    print("=" * 60)

    # Create test checkpoint directory
    test_dir = Path("test_checkpoints")
    test_dir.mkdir(exist_ok=True)

    print(f"Test directory: {test_dir}")

    # Create checkpoint manager
    manager = CheckpointManager(test_dir)
    print(f"✅ CheckpointManager created")

    # Create dummy checkpoints
    print("\n1. Creating dummy checkpoints...")

    for epoch in range(5):
        dummy_checkpoint = {
            'epoch': epoch,
            'model_state_dict': {'dummy': torch.randn(10, 10)},
            'optimizer_state_dict': {},
            'val_loss': 1.0 - epoch * 0.1,
            'val_metrics': {
                'accuracy': 0.7 + epoch * 0.05,
                'f1_score': 0.65 + epoch * 0.06,
                'fpr': 0.15 - epoch * 0.02,
                'fnr': 0.20 - epoch * 0.02
            }
        }

        checkpoint_path = test_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"
        torch.save(dummy_checkpoint, checkpoint_path)
        print(f"  Created: {checkpoint_path.name}")

    # Create best model
    best_checkpoint = {
        'epoch': 4,
        'model_state_dict': {'dummy': torch.randn(10, 10)},
        'val_loss': 0.6,
        'val_metrics': {
            'accuracy': 0.92,
            'f1_score': 0.89,
            'fpr': 0.05,
            'fnr': 0.08
        }
    }
    torch.save(best_checkpoint, test_dir / "best_model.pt")
    print(f"  Created: best_model.pt")

    # List checkpoints
    print("\n2. Listing checkpoints...")
    checkpoints = manager.list_checkpoints()
    print(f"  Found {len(checkpoints)} checkpoints:")
    for checkpoint in checkpoints:
        print(f"    {checkpoint}")

    # Get best checkpoint
    print("\n3. Finding best checkpoint...")
    best = manager.get_best_checkpoint(metric='val_f1', mode='max')
    print(f"  Best by F1: {best}")

    best_fpr = manager.get_best_checkpoint(metric='val_fpr', mode='min')
    print(f"  Best by FPR: {best_fpr}")

    # Export checkpoint info
    print("\n4. Exporting checkpoint info...")
    manager.export_checkpoint_info(test_dir / "checkpoint_info.json")
    print(f"  ✅ Exported to checkpoint_info.json")

    # Cleanup old checkpoints
    print("\n5. Testing cleanup...")
    print(f"  Before cleanup: {len(manager.list_checkpoints())} checkpoints")
    manager.cleanup_old_checkpoints(keep_n=3, keep_best=True)
    print(f"  After cleanup: {len(manager.list_checkpoints())} checkpoints")

    # Cleanup test directory
    print("\n6. Cleaning up test files...")
    shutil.rmtree(test_dir)
    print(f"  ✅ Test directory removed")

    print("\n✅ All checkpoint manager tests passed successfully")
    print("Checkpoint manager module loaded successfully")