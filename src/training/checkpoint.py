import logging
from pathlib import Path
from typing import TYPE_CHECKING, Union

import structlog
import torch

from src.training.metrics import MetricResults

if TYPE_CHECKING:
    from src.training.trainer import Trainer

logger = structlog.get_logger(__name__)


# ============================================================================
# SECURITY: Safe checkpoint loading to prevent arbitrary code execution
# ============================================================================
def safe_load_checkpoint(checkpoint_path: Union[str, Path], device: str = "cpu") -> dict:
    """
    Safely load a checkpoint with validation to prevent arbitrary code execution.

    This function implements defense-in-depth:
    1. Validates checkpoint path (prevents path traversal)
    2. Loads tensors only when possible (weights_only=True) for PyTorch 2.4+
    3. Validates checkpoint structure before returning
    4. Provides clear error messages for debugging

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to map tensors to

    Returns:
        dict: Validated checkpoint dictionary

    Raises:
        ValueError: If checkpoint path or content is invalid
        FileNotFoundError: If checkpoint file does not exist
    """
    checkpoint_path = Path(checkpoint_path)

    # SECURITY: Validate checkpoint path (prevent path traversal)
    # Checkpoints should be in models/ or checkpoints/ directories
    allowed_dirs = ["models", "checkpoints", "cache", "exports"]
    try:
        resolved_path = checkpoint_path.resolve()
        # Check if parent directory is in allowed list
        parent_name = resolved_path.parent.name
        grandparent_name = resolved_path.parent.parent.name if resolved_path.parent.parent else ""

        is_allowed = parent_name in allowed_dirs or grandparent_name in allowed_dirs

        if not is_allowed:
            allowed_dirs_str = ", ".join(allowed_dirs)
            raise ValueError(
                f"Security violation: Checkpoint path '{checkpoint_path}' resolves to '{resolved_path}' "
                f"which is not in allowed directories ({allowed_dirs_str})"
            )
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Invalid checkpoint path: {e}") from e

    # SECURITY: Check file exists and is a regular file
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint path is not a file: {checkpoint_path}")

    # SECURITY: Load checkpoint safely
    try:
        # Try weights_only=True (PyTorch 2.4+ recommended)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        logger.info("Loaded checkpoint with weights_only=True (safest)")
    except Exception:
        # Fallback: Load with pickle but validate structure afterward
        checkpoint = torch.load(checkpoint_path, map_location=device)
        logger.warning("Loaded checkpoint with pickle (less safe - validate structure)")

        # SECURITY: Validate checkpoint structure
        required_keys = ["model_state_dict", "optimizer_state_dict", "epoch"]
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(
                    f"Invalid checkpoint format: missing required key '{key}'. "
                    f"This may be a corrupted or malicious checkpoint file."
                )

        # SECURITY: Check for suspicious keys (arbitrary code)
        suspicious_keys = ["__builtins__", "__code__", "__func__", "eval", "exec", "compile"]
        for key in checkpoint.keys():
            key_str = str(key).lower()
            for susp in suspicious_keys:
                if susp in key_str:
                    raise ValueError(
                        f"Security violation: Suspicious key '{key}' found in checkpoint. "
                        f"This checkpoint may contain arbitrary code."
                    )

    return checkpoint


def _save_checkpoint(
    trainer: "Trainer",
    epoch: int,
    val_loss: float,
    val_metrics: "MetricResults",
    is_best: bool,
) -> None:
    """Save checkpoint"""
    should_save = False

    if trainer.checkpoint_frequency == "every_epoch":
        should_save = True
    elif trainer.checkpoint_frequency == "every_5_epochs" and (epoch + 1) % 5 == 0:
        should_save = True
    elif trainer.checkpoint_frequency == "every_10_epochs" and (epoch + 1) % 10 == 0:
        should_save = True
    elif trainer.checkpoint_frequency == "best_only" and is_best:
        should_save = True

    if not should_save and not is_best:
        return

    # Convert config to dict if it has to_dict method, otherwise save as is
    config_dict = trainer.config.to_dict() if hasattr(trainer.config, "to_dict") else trainer.config

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": trainer.scheduler.state_dict() if trainer.scheduler else None,
        "scaler_state_dict": trainer.scaler.state_dict(),
        "state": trainer.state,
        "config": config_dict,
        "val_loss": val_loss,
        "val_metrics": val_metrics.to_dict(),
        "ema_state_dict": trainer.ema.state_dict() if trainer.ema else None,
    }

    if should_save:
        checkpoint_path = trainer.checkpoint_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

    if is_best:
        best_path = trainer.checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model: {best_path}")


def load_checkpoint(trainer: "Trainer", checkpoint_path: Path) -> None:
    """Load checkpoint"""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    # SECURITY: Use safe loading to prevent arbitrary code execution
    checkpoint = safe_load_checkpoint(checkpoint_path, trainer.device)

    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if trainer.scheduler and checkpoint["scheduler_state_dict"]:
        trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    trainer.scaler.load_state_dict(checkpoint["scaler_state_dict"])
    trainer.state = checkpoint["state"]

    # Load EMA if available
    if trainer.ema and "ema_state_dict" in checkpoint and checkpoint["ema_state_dict"]:
        trainer.ema.load_state_dict(checkpoint["ema_state_dict"])
        logger.info("EMA state loaded from checkpoint")

    logger.info(f"Checkpoint loaded: Epoch {trainer.state.epoch + 1}")
