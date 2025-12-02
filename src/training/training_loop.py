from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from src.training.trainer import Trainer

import structlog
import torch
from tqdm import tqdm

from src.training.metrics import MetricResults
from src.training.optimizer_factory import clip_gradients, get_learning_rate

logger = structlog.get_logger(__name__)


def _run_epoch(
    trainer: "Trainer",
    dataloader: torch.utils.data.DataLoader,
    is_training: bool,
    epoch: int,
) -> Tuple[float, MetricResults]:
    """Run one epoch of training or validation."""
    metrics_tracker = (
        trainer.train_metrics_tracker if is_training else trainer.val_metrics_tracker
    )
    metrics_tracker.reset()

    epoch_loss = 0.0
    num_batches = len(dataloader)

    # Guard against empty dataloaders to avoid division by zero and undefined metrics
    # Earlier versions would crash here when filters yielded no batches, obscuring the
    # real configuration issue and leaving the training run in an inconsistent state.
    if num_batches == 0:
        raise ValueError("Dataloader is empty; cannot run epoch without batches")

    pbar_desc = f"Epoch {epoch+1}/{trainer.config.training.epochs} ["
    pbar_desc += "Train" if is_training else "Val"
    pbar_desc += "]"

    pbar = tqdm(dataloader, desc=pbar_desc, leave=False)

    with torch.set_grad_enabled(is_training):
        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 3:
                inputs, targets, metadata = batch
                # Extract hard negative flags
                # metadata is a dict of lists/tensors
                is_hard_negative = metadata.get("is_hard_negative", None)
                if is_hard_negative is not None:
                    is_hard_negative = is_hard_negative.to(trainer.device, non_blocking=True)
            else:
                inputs, targets = batch
                is_hard_negative = None

            # Move to device
            inputs = inputs.to(trainer.device, non_blocking=True)
            targets = targets.to(trainer.device, non_blocking=True)

            # Keep reference to raw inputs (for distillation)
            raw_inputs = inputs

            # NEW: GPU Processing Pipeline
            # If input is raw audio (B, S) or (B, 1, S), run through AudioProcessor
            if inputs.ndim <= 3:
                trainer.audio_processor.train(is_training)
                inputs = trainer.audio_processor(inputs)
            
            # Apply memory format optimization (now inputs are definitely 4D features)
            inputs = inputs.to(memory_format=torch.channels_last)

            if is_training and trainer.spec_augment is not None:
                inputs = trainer.spec_augment(inputs)

            if is_training:
                trainer.optimizer.zero_grad(set_to_none=True)

            # Disable AMP if QAT is enabled to avoid type mismatch (Float vs Half) in observers
            use_amp = trainer.use_mixed_precision
            if hasattr(trainer.config, "qat") and trainer.config.qat.enabled:
                use_amp = False

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = trainer.model(inputs)
                # Use compute_loss method to allow overriding (e.g., for distillation)
                loss = trainer.compute_loss(
                    outputs,
                    targets,
                    raw_inputs,
                    processed_inputs=inputs,
                    is_hard_negative=is_hard_negative,
                )

            if is_training:
                if use_amp:
                    trainer.scaler.scale(loss).backward()
                    if trainer.gradient_clip > 0:
                        trainer.scaler.unscale_(trainer.optimizer)
                        clip_gradients(trainer.model, trainer.gradient_clip)
                    trainer.scaler.step(trainer.optimizer)
                    trainer.scaler.update()
                else:
                    loss.backward()
                    if trainer.gradient_clip > 0:
                        clip_gradients(trainer.model, trainer.gradient_clip)
                    trainer.optimizer.step()
                
                if trainer.ema is not None:
                    trainer.ema.update()

            metrics_tracker.update(outputs.detach(), targets.detach())
            epoch_loss += loss.item()

            if is_training:
                with torch.no_grad():
                    pred_classes = torch.argmax(outputs, dim=1)
                    batch_acc = (pred_classes == targets).float().mean().item()
                trainer.metric_monitor.update_batch(loss.item(), batch_acc)
                running_avg = trainer.metric_monitor.get_running_averages()
                pbar.set_postfix(
                    {
                        "loss": f"{running_avg['loss']:.4f}",
                        "acc": f"{running_avg['accuracy']:.4f}",
                        "lr": f"{get_learning_rate(trainer.optimizer):.6f}",
                    }
                )
                trainer.state.global_step += 1
                trainer._call_callbacks(
                    "on_batch_end", batch_idx, loss.item(), batch_acc, step=trainer.state.global_step
                )
            else:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = epoch_loss / num_batches
    metrics = metrics_tracker.compute()

    log_prefix = "Train" if is_training else "Val"
    logger.info(f"Epoch {epoch+1} [{log_prefix}]: Loss={avg_loss:.4f}, {metrics}")

    return avg_loss, metrics


def train_epoch(trainer: "Trainer", epoch: int) -> Tuple[float, float]:
    """Train for one epoch"""
    trainer.model.train()
    avg_loss, metrics = _run_epoch(
        trainer, trainer.train_loader, is_training=True, epoch=epoch
    )
    return avg_loss, metrics.accuracy


def validate_epoch(trainer: "Trainer", epoch: int) -> Tuple[float, MetricResults]:
    """Validate for one epoch"""
    trainer.model.eval()
    original_params = None
    if trainer.ema is not None:
        original_params = trainer.ema.apply_shadow()

    try:
        avg_loss, val_metrics = _run_epoch(
            trainer, trainer.val_loader, is_training=False, epoch=epoch
        )
    finally:
        if trainer.ema is not None and original_params is not None:
            trainer.ema.restore(original_params)

    return avg_loss, val_metrics