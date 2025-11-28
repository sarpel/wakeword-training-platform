from pathlib import Path
from typing import TYPE_CHECKING

import structlog
import torch

from src.training.metrics import MetricResults

if TYPE_CHECKING:
    from src.training.trainer import Trainer

logger = structlog.get_logger(__name__)


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
    config_dict = (
        trainer.config.to_dict()
        if hasattr(trainer.config, "to_dict")
        else trainer.config
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": trainer.scheduler.state_dict()
        if trainer.scheduler
        else None,
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

    checkpoint = torch.load(checkpoint_path, map_location=trainer.device)

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
