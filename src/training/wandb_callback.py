import wandb

class WandbCallback:
    """A callback to log training metrics to Weights & Biases."""

    def __init__(self, project_name: str, config: dict):
        """Initializes the WandbCallback.

        Args:
            project_name: The name of the W&B project.
            config: The training configuration.
        """
        wandb.init(project=project_name, config=config)

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float, val_metrics: dict) -> None:
        """Logs metrics at the end of an epoch."""
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_metrics.get("accuracy"),
                "val_f1": val_metrics.get("f1_score"),
                "val_fpr": val_metrics.get("fpr"),
                "val_fnr": val_metrics.get("fnr"),
            }
        )

    def on_batch_end(self, batch_idx: int, loss: float, acc: float) -> None:
        """Logs metrics at the end of a batch."""
        wandb.log({"batch_loss": loss, "batch_acc": acc})
