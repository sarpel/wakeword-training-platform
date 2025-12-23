try:
    import wandb
except ImportError:
    wandb = None  # type: ignore

from typing import Any, Dict, Optional

from src.training.metrics import MetricResults


class WandbCallback:
    """A callback to log training metrics to Weights & Biases."""

    def __init__(self, project_name: str, config: Dict[str, Any]):
        """Initializes the WandbCallback.

        Args:
            project_name: The name of the W&B project.
            config: The training configuration.
        """
        if wandb is None:
            raise ImportError("Weights & Biases is not installed. Please run `pip install wandb`.")

        # Ensure we don't have an active run
        if wandb.run is not None:
            wandb.finish()

        wandb.init(project=project_name, config=config, reinit="finish_previous")

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float, val_metrics: MetricResults) -> None:
        """Logs metrics at the end of an epoch."""
        wandb.log(
            {
                "epoch": epoch + 1,  # Log 1-based epoch
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_metrics.accuracy,
                "val_f1": val_metrics.f1_score,
                "val_fpr": val_metrics.fpr,
                "val_fnr": val_metrics.fnr,
                "val_precision": val_metrics.precision,
                "val_recall": val_metrics.recall,
            }
        )

    def on_batch_end(self, batch_idx: int, loss: float, acc: float, step: Optional[int] = None) -> None:
        """Logs metrics at the end of a batch."""
        log_dict = {"batch_loss": loss, "batch_acc": acc}
        if step is not None:
            # Use the global step as the x-axis for W&B
            wandb.log(log_dict, step=step)
        else:
            wandb.log(log_dict)

    def on_train_end(self) -> None:
        """Called when training finishes."""
        wandb.finish()
