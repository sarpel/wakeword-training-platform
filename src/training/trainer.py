"""
Wakeword Training Loop
GPU-accelerated training with checkpointing, early stopping, and metrics tracking
"""

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

if TYPE_CHECKING:
    from src.config.defaults import WakewordConfig

import structlog
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config.cuda_utils import enforce_cuda
from src.config.seed_utils import set_seed
from src.data.augmentation import SpecAugment
from src.data.processor import AudioProcessor  # FIXED
from src.models.losses import create_loss_function
from src.training.checkpoint_manager import CheckpointManager
from src.training.ema import EMA, EMAScheduler
from src.training.metrics import MetricMonitor, MetricResults, MetricsTracker
from src.training.optimizer_factory import create_grad_scaler, create_optimizer_and_scheduler, get_learning_rate
from src.training.training_loop import train_epoch, validate_epoch

logger = structlog.get_logger(__name__)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


@dataclass
class TrainingState:
    """Container for training state"""

    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    best_val_f1: float = -1.0
    best_val_fpr: float = 1.0
    best_val_pauc: float = 0.0
    epochs_without_improvement: int = 0
    training_time: float = 0.0
    should_stop: bool = False  # Added external stop flag


class Trainer:
    """
    Main training loop for wakeword detection
    GPU-accelerated with comprehensive metrics tracking
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: "WakewordConfig",
        checkpoint_manager: CheckpointManager,
        device: str = "cuda",
        use_ema: Optional[bool] = None,
        ema_decay: Optional[float] = None,
    ) -> None:
        """
        Initialize trainer

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            checkpoint_manager: Checkpoint manager
            device: Device for training
            use_ema: Whether to use Exponential Moving Average (overrides config)
            ema_decay: EMA decay rate (overrides config)
        """
        # Enforce GPU requirement (but allow CPU if needed/configured)
        enforce_cuda(allow_cpu=True)

        self.device = device
        self.config = config

        # Determine EMA settings (Argument > Config > Default)
        if use_ema is not None:
            self.use_ema = use_ema
        else:
            self.use_ema = getattr(config.training, "use_ema", True)

        if ema_decay is not None:
            ema_decay = ema_decay
        else:
            ema_decay = getattr(config.training, "ema_decay", 0.999)

        ema_final_decay = getattr(config.training, "ema_final_decay", 0.9995)
        ema_final_epochs = getattr(config.training, "ema_final_epochs", 10)
        metric_window_size = getattr(config.training, "metric_window_size", 100)

        # External stop control
        self.stop_event = threading.Event()

        # Move model to GPU
        self.model = model.to(device)
        # channels_last bellek düzeni (Ampere+ için throughput ↑) - Only on CUDA
        if device == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)  # type: ignore
            logger.info("Using channels_last memory format for model")

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # NEW: Audio Processor for GPU-based augmentation/feature extraction
        from src.config.paths import paths

        cmvn_path = paths.CMVN_STATS
        self.audio_processor = AudioProcessor(
            config=config, cmvn_path=cmvn_path if cmvn_path.exists() else None, device=device
        )

        # Calculate class weights
        class_weights = None
        if hasattr(train_loader.dataset, "get_class_weights"):
            try:
                # Calculate weights from training data
                class_weights = train_loader.dataset.get_class_weights()
                logger.info(f"Class weights calculated: {class_weights}")
            except Exception as e:
                logger.warning(f"Failed to calculate class weights: {e}")

        # Create loss function
        # Initialize with reduction='none' to support hard negative weighting
        self.criterion = create_loss_function(
            loss_name=config.loss.loss_function,
            num_classes=config.model.num_classes,
            label_smoothing=config.loss.label_smoothing,
            focal_alpha=config.loss.focal_alpha,
            focal_gamma=config.loss.focal_gamma,
            class_weights=class_weights,
            device=device,
            reduction="none",  # Changed from default 'mean'
        ).to(device)

        # Create optimizer and scheduler (self.model ile kur)
        self.optimizer, self.scheduler = create_optimizer_and_scheduler(
            self.model, config, steps_per_epoch=len(train_loader)
        )

        # Mixed precision training (Only on CUDA)
        self.use_mixed_precision = config.optimizer.mixed_precision and device == "cuda"
        self.scaler = create_grad_scaler(enabled=self.use_mixed_precision)

        # Gradient clipping
        self.gradient_clip = config.optimizer.gradient_clip

        # Metrics tracking
        self.train_metrics_tracker = MetricsTracker(device=device)
        self.val_metrics_tracker = MetricsTracker(device=device)
        self.metric_monitor = MetricMonitor(window_size=metric_window_size)

        # Training state
        self.state = TrainingState()

        # Early stopping
        self.early_stopping_patience = config.training.early_stopping_patience

        self.checkpoint_manager = checkpoint_manager
        self.checkpoint_dir = checkpoint_manager.checkpoint_dir
        self.checkpoint_frequency = config.training.checkpoint_frequency

        # Callbacks
        self.callbacks: List[Any] = []

        # SpecAugment (GPU-based, applied during training only)
        self.use_spec_augment = getattr(config.augmentation, "use_spec_augment", True)
        self.spec_augment: Optional[SpecAugment] = None
        if self.use_spec_augment:
            self.spec_augment = SpecAugment(
                freq_mask_param=getattr(config.augmentation, "freq_mask_param", 15),
                time_mask_param=getattr(config.augmentation, "time_mask_param", 30),
                n_freq_masks=getattr(config.augmentation, "n_freq_masks", 2),
                n_time_masks=getattr(config.augmentation, "n_time_masks", 2),
            )
            logger.info("SpecAugment initialized (GPU-based)")

        # EMA (Exponential Moving Average)
        self.ema = None
        self.ema_scheduler = None
        if self.use_ema:
            self.ema = EMA(self.model, decay=ema_decay)
            self.ema_scheduler = EMAScheduler(
                self.ema,
                initial_decay=ema_decay,
                final_decay=ema_final_decay,
                warmup_epochs=0,
                final_epochs=ema_final_epochs,
            )
            logger.info(f"EMA initialized with decay={ema_decay}")

        logger.info("Trainer initialized")
        logger.info(f"  Device: {device}")
        logger.info(f"  Model: {config.model.architecture}")
        logger.info(f"  Optimizer: {config.optimizer.optimizer}")
        logger.info(f"  Scheduler: {config.optimizer.scheduler}")
        logger.info(f"  Loss: {config.loss.loss_function}")
        logger.info(f"  Mixed precision: {self.use_mixed_precision}")
        logger.info(f"  Gradient clipping: {self.gradient_clip}")
        logger.info(f"  Early stopping patience: {self.early_stopping_patience}")
        logger.info(f"  Checkpoint dir: {self.checkpoint_dir}")
        logger.info(f"  SpecAugment: {self.use_spec_augment}")
        logger.info(f"  EMA: {self.use_ema}")

    def train(
        self,
        start_epoch: int = 0,
        resume_from: Optional[Path] = None,
        seed: int = 42,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """
        Full training loop

        Args:
            start_epoch: Starting epoch number
            resume_from: Path to checkpoint to resume from
            seed: Random seed for reproducibility
            deterministic: If True, enables deterministic mode (slower but reproducible)
        """
        # Set seed for reproducibility
        set_seed(seed, deterministic=deterministic)

        if resume_from is not None:
            self.checkpoint_manager.load_checkpoint(resume_from, self.model, self.optimizer, self.device)
            start_epoch = self.state.epoch + 1
            logger.info(f"Resumed from checkpoint at epoch {start_epoch}")

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "val_fpr": [],
            "val_fnr": [],
            "val_pauc": [],
            "learning_rates": [],
        }

        logger.info("=" * 80)
        logger.info("STARTING TRAINING SESSION")
        logger.info(f"  Epochs: {self.config.training.epochs}")
        logger.info(f"  Training batches: {len(self.train_loader)}")
        logger.info(f"  Validation batches: {len(self.val_loader)}")
        logger.info(f"  Batch size: {self.config.training.batch_size}")
        logger.info("=" * 80)

        start_time = time.time()

        try:
            for epoch in range(start_epoch, self.config.training.epochs):
                logger.info(f"Epoch {epoch+1}/{self.config.training.epochs} starting")
                self._call_callbacks("on_epoch_start", epoch)

                # Check for QAT transition
                self._check_qat_transition(epoch)

                try:
                    train_loss, train_acc = train_epoch(self, epoch)
                    val_loss, val_metrics = validate_epoch(self, epoch)
                except ConnectionResetError:
                    if sys.platform == "win32":
                        logger.warning("ConnectionResetError suppressed during training loop (harmless Windows error)")
                        # Try to continue or return current state
                        continue
                    else:
                        raise

                self._update_scheduler(val_loss)
                current_lr = get_learning_rate(self.optimizer)

                # Update EMA decay schedule
                if self.ema_scheduler is not None:
                    ema_decay = self.ema_scheduler.step(epoch, self.config.training.epochs)
                    logger.debug(f"EMA decay updated to {ema_decay:.5f}")

                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_metrics.accuracy)
                history["val_f1"].append(val_metrics.f1_score)
                history["val_fpr"].append(val_metrics.fpr)
                history["val_fnr"].append(val_metrics.fnr)
                history["val_pauc"].append(val_metrics.pauc)
                history["learning_rates"].append(current_lr)

                self.val_metrics_tracker.save_epoch_metrics(val_metrics)

                improved = self._check_improvement(val_loss, val_metrics)

                self.checkpoint_manager.save_checkpoint(self, epoch, val_loss, val_metrics, improved)

                if self._should_stop_early():
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

                if self.stop_event.is_set():
                    logger.info(f"Training stopped by user after {epoch+1} epochs")
                    break

                self._call_callbacks("on_epoch_end", epoch, train_loss, val_loss, val_metrics)

                logger.info(f"Epoch {epoch+1}/{self.config.training.epochs} Results:")
                logger.info(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
                logger.info(
                    f"  Val:   Loss={val_loss:.4f}, Acc={val_metrics.accuracy:.4f}, "
                    f"F1={val_metrics.f1_score:.4f}, pAUC={val_metrics.pauc:.4f}, FPR={val_metrics.fpr:.4f}, FNR={val_metrics.fnr:.4f}"
                )

                # Standout F1 Score log
                f1_formatted = f"{val_metrics.f1_score:.3f}".replace(".", ",")
                logger.info("\n" + "-" * 30)
                logger.info(f"⭐ F1 SCORE: {f1_formatted} ⭐")
                logger.info("-" * 30 + "\n")

                logger.info(f"  Learning Rate: {current_lr:.6f}")
                if improved:
                    logger.info(f"  ✅ New best model saved! (F1: {val_metrics.f1_score:.4f})")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        end_time = time.time()
        self.state.training_time = end_time - start_time

        logger.info("=" * 80)
        logger.info("TRAINING SESSION COMPLETE")
        logger.info(f"  Total time: {self.state.training_time / 3600:.2f} hours")
        logger.info(f"  Best val loss: {self.state.best_val_loss:.4f}")
        logger.info(f"  Best val F1: {self.state.best_val_f1:.4f}")
        logger.info(f"  Best val FPR: {self.state.best_val_fpr:.4f}")
        logger.info("=" * 80)

        best_f1_epoch, best_f1_metrics = self.val_metrics_tracker.get_best_epoch("f1_score")
        best_fpr_epoch, best_fpr_metrics = self.val_metrics_tracker.get_best_epoch("fpr")

        if best_f1_metrics:
            logger.info(f"\n🏆 BEST F1 SCORE: {best_f1_metrics.f1_score:.4f} ⭐ (Epoch {best_f1_epoch+1})")
        if best_fpr_metrics:
            logger.info(f"BEST FPR: {best_fpr_metrics.fpr:.4f} (Epoch {best_fpr_epoch+1})")

        results = {
            "history": history,
            "final_epoch": self.state.epoch,
            "best_val_loss": self.state.best_val_loss,
            "best_val_f1": self.state.best_val_f1,
            "best_val_fpr": self.state.best_val_fpr,
            "training_time": self.state.training_time,
            "best_f1_epoch": best_f1_epoch,
            "best_fpr_epoch": best_fpr_epoch,
        }

        # Final QAT reporting if enabled
        if getattr(self.config.qat, "enabled", False):
            try:
                from src.training.qat_utils import compare_model_accuracy, convert_model_to_quantized

                logger.info("Generating Quantization Error Report...")
                # We need a copy of the model to convert it to quantized without destroying the original
                # However, for the report, we can just convert it since training is done.
                fp32_model = self.model  # This is actually the QAT model (with fake quants)

                # To get true INT8 performance, we convert
                # Moving to CPU first is safer for quantization conversion in some PyTorch versions
                self.model.to("cpu")
                quantized_model = convert_model_to_quantized(self.model)

                qat_report = compare_model_accuracy(
                    fp32_model, quantized_model, self.val_loader, device="cpu", audio_processor=self.audio_processor
                )
                results["qat_report"] = qat_report
            except Exception as e:
                logger.warning(f"Failed to generate QAT report: {e}")

        self._call_callbacks("on_train_end")

        return results

    def _check_qat_transition(self, epoch: int) -> None:
        """Check and handle transition to QAT fine-tuning"""
        if not getattr(self.config.qat, "enabled", False):
            return

        if epoch == self.config.qat.start_epoch:
            logger.info(f"--- QAT Transition Triggered (Epoch {epoch}) ---")

            # 1. Prepare model for QAT (inserts observers)
            from src.training.qat_utils import prepare_model_for_qat

            self.model.train()
            self.model = prepare_model_for_qat(self.model, self.config.qat)
            self.model.to(self.device)

            # 2. Calibrate model to initialize observers with statistics
            from src.training.qat_utils import calibrate_model

            logger.info("Calibrating QAT observers...")
            calibrate_model(self.model, self.val_loader, device=self.device, audio_processor=self.audio_processor)

            # 3. Re-initialize optimizer for the new model parameters (fake quants)
            # Use lower learning rate for QAT fine-tuning if desired
            self.optimizer, self.scheduler = create_optimizer_and_scheduler(
                self.model, self.config, steps_per_epoch=len(self.train_loader)
            )

            # 4. Disable mixed precision as it can interfere with QAT observers
            if self.use_mixed_precision:
                logger.info("Disabling mixed precision for QAT fine-tuning")
                self.use_mixed_precision = False
                self.scaler = create_grad_scaler(enabled=False)

    def _update_scheduler(self, val_loss: float) -> None:
        """Update learning rate scheduler"""
        if self.scheduler is not None:
            if hasattr(self.scheduler, "step"):
                try:
                    self.scheduler.step(val_loss)
                except TypeError:
                    self.scheduler.step()

    def _check_improvement(self, val_loss: float, val_metrics: MetricResults) -> bool:
        """Check if model improved based on primary metric (val_f1)
        Simplified to use single primary metric for early stopping
        """
        improved = False
        val_f1 = val_metrics.f1_score
        val_fpr = val_metrics.fpr
        val_pauc = val_metrics.pauc

        # Track all metrics for logging
        if val_loss < self.state.best_val_loss:
            self.state.best_val_loss = val_loss

        if val_fpr < self.state.best_val_fpr and self.state.epoch > 0:
            self.state.best_val_fpr = val_fpr

        if val_pauc > self.state.best_val_pauc:
            self.state.best_val_pauc = val_pauc

        # Primary metric: val_f1 (for early stopping)
        if val_f1 > self.state.best_val_f1:
            self.state.best_val_f1 = val_f1
            improved = True
            self.state.epochs_without_improvement = 0
        else:
            self.state.epochs_without_improvement += 1

        return improved

    def _should_stop_early(self) -> bool:
        """Check if training should stop early"""
        return self.state.epochs_without_improvement >= self.early_stopping_patience

    def add_callback(self, callback: Any) -> None:
        """Add training callback"""
        self.callbacks.append(callback)

    def _call_callbacks(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Call all callbacks for event"""
        for callback in self.callbacks:
            if hasattr(callback, event):
                getattr(callback, event)(*args, **kwargs)

    def stop(self) -> None:
        """Signal the trainer to stop at the next available opportunity"""
        self.stop_event.set()

    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        inputs: Optional[torch.Tensor] = None,
        processed_inputs: Optional[torch.Tensor] = None,
        is_hard_negative: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute loss. Can be overridden for distillation or custom logic.

        Args:
            outputs: Model predictions
            targets: Ground truth labels
            inputs: Original raw inputs (optional, for distillation/teacher)
            processed_inputs: Processed inputs/features (optional, for embedding extraction)
            is_hard_negative: Tensor indicating hard negative samples (1=hard, 0=normal)

        Returns:
            Loss tensor
        """
        # Check if using TripletLoss
        from src.models.losses import TripletLoss

        if isinstance(self.criterion, TripletLoss):
            # For TripletLoss, we need embeddings, not logits.
            # Re-compute embeddings using processed_inputs if available.
            if processed_inputs is not None and hasattr(self.model, "embed"):
                embeddings = self.model.embed(processed_inputs)
                return cast(torch.Tensor, self.criterion(embeddings, targets))
            else:
                # Fallback: assume outputs are embeddings or model doesn't support embed()
                # This might happen if using a model without embed() method
                return cast(torch.Tensor, self.criterion(outputs, targets))

        # Standard loss (CrossEntropy, FocalLoss)
        # Criterion is initialized with reduction='none', so it returns (B,)
        loss = self.criterion(outputs, targets)

        # Apply hard negative weighting
        if is_hard_negative is not None:
            # Get weight from config (default 1.5 if not set)
            hn_weight = getattr(self.config.loss, "hard_negative_weight", 1.5)

            # Create weight tensor: 1.0 for normal, hn_weight for hard negatives
            # is_hard_negative is 1 for hard negatives, 0 otherwise
            weights = torch.ones_like(loss)
            weights = weights + (is_hard_negative * (hn_weight - 1.0))

            loss = loss * weights

        # Return mean loss
        return loss.mean()


if __name__ == "__main__":
    # Test trainer initialization
    print("Trainer Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cpu":
        print("⚠️  CUDA not available - trainer requires GPU")
        print("This is a basic initialization test only")

    # Create dummy model
    from src.models.architectures import create_model

    model = create_model("resnet18", num_classes=2, pretrained=False)
    print("✅ Created model: ResNet18")

    # Create dummy config
    from src.config.defaults import WakewordConfig

    config = WakewordConfig()
    print("✅ Created config")

    # Create dummy data loaders
    dummy_dataset = torch.utils.data.TensorDataset(
        torch.randn(100, 1, 64, 50),  # Spectrograms
        torch.randint(0, 2, (100,)),  # Labels
    )

    train_loader = DataLoader(dummy_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(dummy_dataset, batch_size=8, shuffle=False)

    print(f"✅ Created data loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")

    # Create dummy checkpoint manager
    from src.training.checkpoint_manager import CheckpointManager

    checkpoint_manager = CheckpointManager(checkpoint_dir=Path("test_checkpoints"))

    # Create trainer
    try:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            checkpoint_manager=checkpoint_manager,
            device=device,
        )
        print("✅ Trainer initialized successfully")

        print("\n✅ Trainer module loaded successfully")
        print("Note: Full training test requires actual dataset and GPU")

    except SystemExit:
        print("\n❌ Trainer requires CUDA GPU (as specified in requirements)")
        print("  This is expected behavior - CPU fallback not allowed")
