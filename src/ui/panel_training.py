"""
Panel 3: Model Training
- Start/pause/stop training with async execution
- Live metrics display
- Real-time plotting
- GPU monitoring
- Training state management
"""

import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")  # Non-interactive backend
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

from src.config.cuda_utils import get_cuda_validator
from src.config.defaults import WakewordConfig
from src.config.paths import paths  # NEW: Centralized paths
from src.data.balanced_sampler import create_balanced_sampler_from_dataset
from src.data.cmvn import compute_cmvn_from_dataset
from src.data.dataset import WakewordDataset, load_dataset_splits
from src.data.processor import AudioProcessor
from src.exceptions import WakewordException
from src.models.architectures import create_model
from src.training.checkpoint_manager import CheckpointManager
from src.training.distillation_trainer import DistillationTrainer
from src.training.hpo import run_hpo
from src.training.lr_finder import LRFinder
from src.training.metrics import MetricResults  # Imported MetricResults
from src.training.qat_utils import prepare_model_for_qat
from src.training.trainer import Trainer
from src.training.wandb_callback import WandbCallback


class TrainingState:
    """Global training state manager"""

    def __init__(self) -> None:
        self.is_training = False
        self.should_stop = False
        self.should_pause = False
        self.is_paused = False

        self.trainer: Optional[Trainer] = None
        self.config: Optional[WakewordConfig] = None
        self.model: Optional[torch.nn.Module] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.training_thread: Optional[threading.Thread] = None

        # Metrics history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "val_fpr": [],
            "val_fnr": [],
            "epochs": [],
        }

        # Current metrics
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_batch = 0
        self.total_batches = 0
        self.current_train_loss = 0.0
        self.current_train_acc = 0.0
        self.current_val_loss = 0.0
        self.current_val_acc = 0.0
        self.current_fpr = 0.0
        self.current_fnr = 0.0
        self.current_speed = 0.0
        self.eta_seconds = 0

        # Best metrics
        self.best_epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_f1 = 0.0
        self.best_val_acc = 0.0
        self.best_model_path = "No model saved yet"

        # Training thread
        self.training_thread = None
        self.hpo_thread = None
        self.log_queue: queue.Queue[str] = queue.Queue()

    def reset(self) -> None:
        """Reset state for new training"""
        self.is_training = False
        self.should_stop = False
        self.should_pause = False
        self.is_paused = False
        self.current_epoch = 0
        self.current_batch = 0
        self.eta_seconds = 0

    def add_log(self, message: str) -> None:
        """Add message to log queue"""
        self.log_queue.put(f"[{time.strftime('%H:%M:%S')}] {message}\n")
        print(message)


# Global training state
training_state = TrainingState()


def create_loss_plot() -> plt.Figure:
    """Create loss curve plot"""
    fig, ax = plt.subplots(figsize=(10, 5))

    if len(training_state.history["epochs"]) > 0:
        epochs = training_state.history["epochs"]

        ax.plot(
            epochs,
            training_state.history["train_loss"],
            label="Train Loss",
            marker="o",
            linewidth=2,
        )
        ax.plot(
            epochs,
            training_state.history["val_loss"],
            label="Val Loss",
            marker="s",
            linewidth=2,
        )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No data yet",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig


def create_accuracy_plot() -> plt.Figure:
    """Create accuracy curve plot"""
    fig, ax = plt.subplots(figsize=(10, 5))

    if len(training_state.history["epochs"]) > 0:
        epochs = training_state.history["epochs"]

        ax.plot(
            epochs,
            [a * 100 for a in training_state.history["train_acc"]],
            label="Train Acc",
            marker="o",
            linewidth=2,
        )
        ax.plot(
            epochs,
            [a * 100 for a in training_state.history["val_acc"]],
            label="Val Acc",
            marker="s",
            linewidth=2,
        )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
    else:
        ax.text(
            0.5,
            0.5,
            "No data yet",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig


def create_metrics_plot() -> plt.Figure:
    """Create FPR/FNR plot"""
    fig, ax = plt.subplots(figsize=(10, 5))

    if len(training_state.history["epochs"]) > 0:
        epochs = training_state.history["epochs"]

        ax.plot(
            epochs,
            [f * 100 for f in training_state.history["val_fpr"]],
            label="FPR",
            marker="o",
            linewidth=2,
            color="red",
        )
        ax.plot(
            epochs,
            [f * 100 for f in training_state.history["val_fnr"]],
            label="FNR",
            marker="s",
            linewidth=2,
            color="orange",
        )
        ax.plot(
            epochs,
            [f * 100 for f in training_state.history["val_f1"]],
            label="F1 Score",
            marker="^",
            linewidth=2,
            color="green",
        )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Rate (%)", fontsize=12)
        ax.set_title("Validation Metrics (FPR, FNR, F1)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No data yet",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title("Validation Metrics", fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig


def format_time(seconds: float) -> str:
    """Format seconds to HH:MM:SS"""
    if seconds <= 0 or seconds == float("inf"):
        return "--:--:--"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def training_worker() -> None:
    """Background thread for training"""
    try:
        if training_state.config is None or training_state.trainer is None:
            training_state.add_log("ERROR: Config or Trainer not initialized")
            return

        training_state.add_log("Starting training...")
        training_state.add_log(f"Configuration: {training_state.config.config_name}")
        training_state.add_log(f"Model: {training_state.config.model.architecture}")
        training_state.add_log(f"Epochs: {training_state.config.training.epochs}")
        training_state.add_log(f"Batch size: {training_state.config.training.batch_size}")
        training_state.add_log("-" * 60)

        # Capture trainer locally for type safety
        trainer = training_state.trainer

        # Create custom callback for live updates
        class LiveUpdateCallback:
            def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float, val_metrics: MetricResults) -> None:
                # Update state
                training_state.current_epoch = epoch + 1
                training_state.current_train_loss = train_loss
                training_state.current_val_loss = val_loss
                training_state.current_val_acc = val_metrics.accuracy
                training_state.current_fpr = val_metrics.fpr
                training_state.current_fnr = val_metrics.fnr

                # Update history
                training_state.history["epochs"].append(epoch + 1)
                training_state.history["train_loss"].append(train_loss)
                training_state.history["train_acc"].append(training_state.current_train_acc)
                training_state.history["val_loss"].append(val_loss)
                training_state.history["val_acc"].append(val_metrics.accuracy)
                training_state.history["val_f1"].append(val_metrics.f1_score)
                training_state.history["val_fpr"].append(val_metrics.fpr)
                training_state.history["val_fnr"].append(val_metrics.fnr)

                # Update best metrics
                if val_loss < training_state.best_val_loss:
                    training_state.best_val_loss = val_loss

                if val_metrics.f1_score > training_state.best_val_f1:
                    training_state.best_val_f1 = val_metrics.f1_score
                    training_state.best_epoch = epoch + 1
                    training_state.best_model_path = str(trainer.checkpoint_dir / "best_model.pt")

                if val_metrics.accuracy > training_state.best_val_acc:
                    training_state.best_val_acc = val_metrics.accuracy

                # Log
                training_state.add_log(
                    f"Epoch {epoch+1}/{training_state.total_epochs} - "
                    f"Loss: {train_loss:.4f}/{val_loss:.4f} - "
                    f"Acc: {training_state.current_train_acc:.2%}/{val_metrics.accuracy:.2%} - "
                    f"FPR: {val_metrics.fpr:.2%} - FNR: {val_metrics.fnr:.2%}"
                )

            def on_batch_end(self, batch_idx: int, loss: float, acc: float, **kwargs: Any) -> None:
                training_state.current_batch = batch_idx + 1
                training_state.current_train_loss = loss
                training_state.current_train_acc = acc

        # Add callback
        callback = LiveUpdateCallback()
        trainer.add_callback(callback)

        # Train
        start_time = time.time()
        results = trainer.train()
        elapsed = time.time() - start_time

        # Training complete
        training_state.add_log("-" * 60)
        training_state.add_log(f"Training complete!")
        training_state.add_log(f"Total time: {elapsed/3600:.2f} hours")
        training_state.add_log(f"Best epoch: {training_state.best_epoch}")
        training_state.add_log(f"Best val F1: {training_state.best_val_f1:.4f}")
        training_state.add_log(f"Best val loss: {training_state.best_val_loss:.4f}")
        training_state.add_log(f"Best val acc: {training_state.best_val_acc:.2%}")
        training_state.add_log(f"Model saved to: {training_state.best_model_path}")

    except WakewordException as e:
        training_state.add_log(
            f"ERROR: {str(e)}\n\nActionable suggestion: Please check your configuration and data for the following error: {e}"
        )
        logger.exception("Training failed")
    except Exception as e:
        training_state.add_log(f"ERROR: {str(e)}")
        logger.exception("Training failed")
    finally:
        training_state.is_training = False
        training_state.train_loader = None
        training_state.val_loader = None
        torch.cuda.empty_cache()


def start_training(
    config_state: Dict,
    use_cmvn: bool,
    use_ema: bool,
    ema_decay: float,
    use_balanced_sampler: bool,
    sampler_ratio_pos: int,
    sampler_ratio_neg: int,
    sampler_ratio_hard: int,
    run_lr_finder: bool,
    use_wandb: bool,
    wandb_project: str,
    loss_func_name: str,
    loss_smoothing: float,
    focal_gamma: float,
    focal_alpha: float,
    hard_neg_weight: float,
    wandb_api_key: str = "",
    resume_checkpoint: Optional[str] = None,
) -> Tuple:
    """Start training with current configuration and advanced features"""
    if training_state.is_training:
        return (
            "⚠️ Training already in progress",
            f"{training_state.current_epoch}/{training_state.total_epochs}",
            f"{training_state.current_batch}/{training_state.total_batches}",
            None,
            None,
            None,
        )

    try:
        # Handle Resume
        resume_path = None
        if resume_checkpoint and resume_checkpoint != "None":
            resume_path = paths.CHECKPOINTS / resume_checkpoint
            if not resume_path.exists():
                return (
                    f"❌ Checkpoint not found: {resume_checkpoint}",
                    "0/0",
                    "0/0",
                    None,
                    None,
                    None,
                )
            training_state.add_log(f"Resuming from checkpoint: {resume_checkpoint}")
        # Handle W&B Login if key provided
        if use_wandb and wandb_api_key.strip():
            try:
                import wandb

                training_state.add_log(f"Logging into W&B...")
                wandb.login(key=wandb_api_key.strip())
                save_wandb_key(wandb_api_key.strip())  # Save key on successful use
                training_state.add_log("✅ W&B Login successful")
            except Exception as e:
                training_state.add_log(f"⚠️ W&B Login failed: {e}")
                # We don't return here, we let it try to continue or fail later if critical
        # Get config from global state
        if "config" not in config_state or config_state["config"] is None:
            return (
                "❌ No configuration loaded. Please configure in Panel 2 first.",
                "0/0",
                "0/0",
                None,
                None,
                None,
            )

        config = config_state["config"]
        # Update loss config
        config.loss.loss_function = loss_func_name
        config.loss.label_smoothing = loss_smoothing
        config.loss.focal_gamma = focal_gamma
        config.loss.focal_alpha = focal_alpha
        config.loss.hard_negative_weight = hard_neg_weight
        
        training_state.config = config
        training_state.total_epochs = config.training.epochs

        training_state.add_log("Initializing training...")

        # Check if dataset splits exist
        # Use centralized paths
        splits_dir = paths.SPLITS
        if not splits_dir.exists() or not (splits_dir / "train.json").exists():
            return (
                "❌ Dataset splits not found. Please run Panel 1 to scan and split datasets first.",
                "0/0",
                "0/0",
                None,
                None,
                None,
            )

        training_state.add_log("Loading datasets...")

        # Load datasets
        aug_config = {
            "time_stretch_range": (
                config.augmentation.time_stretch_min,
                config.augmentation.time_stretch_max,
            ),
            "pitch_shift_range": (
                config.augmentation.pitch_shift_min,
                config.augmentation.pitch_shift_max,
            ),
            "background_noise_prob": config.augmentation.background_noise_prob,
            "noise_snr_range": (
                config.augmentation.noise_snr_min,
                config.augmentation.noise_snr_max,
            ),
            "rir_prob": config.augmentation.rir_prob,
            "time_shift_prob": getattr(config.augmentation, "time_shift_prob", 0.0),
            "time_shift_range_ms": (
                getattr(config.augmentation, "time_shift_min_ms", -100),
                getattr(config.augmentation, "time_shift_max_ms", 100),
            ),
        }

        # Normalize feature type name
        feature_type = "mel" if config.data.feature_type == "mel_spectrogram" else config.data.feature_type

        # Handle CMVN
        cmvn_path = None
        if use_cmvn:
            cmvn_path = paths.CMVN_STATS
            
            # Check if we need to recompute stats (missing or dimension mismatch)
            should_compute_cmvn = False
            expected_dim = config.data.n_mels if feature_type == "mel" else config.data.n_mfcc
            
            if not cmvn_path.exists():
                should_compute_cmvn = True
            else:
                try:
                    import json
                    with open(cmvn_path, "r") as f:
                        stats = json.load(f)
                    loaded_dim = len(stats["mean"])
                    if loaded_dim != expected_dim:
                        training_state.add_log(f"⚠️ CMVN stats dimension mismatch (Found {loaded_dim}, Expected {expected_dim}). Recomputing...")
                        should_compute_cmvn = True
                except Exception as e:
                    training_state.add_log(f"⚠️ Failed to verify CMVN stats: {e}. Recomputing...")
                    should_compute_cmvn = True

            if should_compute_cmvn:
                training_state.add_log("Computing CMVN statistics (this may take a moment)...")
                # Load datasets temporarily without CMVN to compute stats
                temp_train_ds, _, _ = load_dataset_splits(
                    data_root=paths.DATA,
                    device="cuda",
                    feature_type=feature_type,
                    n_mels=config.data.n_mels,
                    n_mfcc=config.data.n_mfcc,
                    n_fft=config.data.n_fft,
                    hop_length=config.data.hop_length,
                    use_precomputed_features_for_training=config.data.use_precomputed_features_for_training,
                    fallback_to_audio=True,  # Force fallback to avoid shape mismatch crashes during CMVN
                    apply_cmvn=False,
                )
                compute_cmvn_from_dataset(temp_train_ds, cmvn_path, max_samples=1000)
                training_state.add_log(f"✅ CMVN stats saved to {cmvn_path}")

        training_state.add_log("IMPORTANT: Forcing augmentation for training by disabling precomputed features.")

        # NEW: Optimize config for GPU pipeline
        import os

        # Use min(16, cpu_count) for workers
        optimal_workers = min(16, os.cpu_count() or 1)
        config.training.num_workers = optimal_workers
        config.training.pin_memory = True

        # Prefetch factor (only if workers > 0)
        prefetch_factor = 4 if optimal_workers > 0 else None

        train_ds = WakewordDataset(
            manifest_path=splits_dir / "train.json",
            sample_rate=config.data.sample_rate,
            audio_duration=config.data.audio_duration,
            augment=True,
            augmentation_config=aug_config,
            background_noise_dir=paths.BACKGROUND_NOISE,
            rir_dir=paths.RIRS,
            device="cuda",
            feature_type=feature_type,
            n_mels=config.data.n_mels,
            n_mfcc=config.data.n_mfcc,
            n_fft=config.data.n_fft,
            hop_length=config.data.hop_length,
            use_precomputed_features_for_training=False,  # Force augmentation
            npy_cache_features=config.data.npy_cache_features,
            fallback_to_audio=True,  # IMPORTANT
            cmvn_path=cmvn_path,
            apply_cmvn=use_cmvn,
            return_raw_audio=True,  # NEW: Use GPU pipeline
        )

        val_ds = WakewordDataset(
            manifest_path=splits_dir / "val.json",
            sample_rate=config.data.sample_rate,
            audio_duration=config.data.audio_duration,
            augment=False,
            device="cuda",
            feature_type=feature_type,
            n_mels=config.data.n_mels,
            n_mfcc=config.data.n_mfcc,
            n_fft=config.data.n_fft,
            hop_length=config.data.hop_length,
            use_precomputed_features_for_training=config.data.use_precomputed_features_for_training,
            npy_cache_features=config.data.npy_cache_features,
            fallback_to_audio=True,  # FORCE TRUE to handle shape mismatches automatically
            cmvn_path=cmvn_path,
            apply_cmvn=use_cmvn,
            return_raw_audio=True,  # NEW: Use GPU pipeline
        )

        # test_ds is not used in training, so we don't load it here to save memory
        test_ds = None

        training_state.add_log(f"Loaded {len(train_ds)} training samples")
        training_state.add_log(f"Loaded {len(val_ds)} validation samples")

        if use_cmvn:
            training_state.add_log("✅ CMVN normalization enabled")

        # Create data loaders with optional balanced sampling
        if use_balanced_sampler:
            training_state.add_log("Creating balanced batch sampler...")
            try:
                train_sampler = create_balanced_sampler_from_dataset(
                    dataset=train_ds,
                    batch_size=config.training.batch_size,
                    ratio=(sampler_ratio_pos, sampler_ratio_neg, sampler_ratio_hard),
                    drop_last=True,
                )
                training_state.train_loader = DataLoader(
                    train_ds,
                    batch_sampler=train_sampler,
                    num_workers=config.training.num_workers,
                    pin_memory=True,
                    persistent_workers=True if config.training.num_workers > 0 else False,
                    prefetch_factor=prefetch_factor,
                )
                training_state.add_log(
                    f"✅ Balanced sampler enabled (ratio {sampler_ratio_pos}:{sampler_ratio_neg}:{sampler_ratio_hard})"
                )
            except Exception as e:
                training_state.add_log(f"⚠️ Balanced sampler failed: {e}")
                training_state.add_log("Falling back to standard DataLoader...")
                training_state.train_loader = DataLoader(
                    train_ds,
                    batch_size=config.training.batch_size,
                    shuffle=True,
                    num_workers=config.training.num_workers,
                    pin_memory=True,
                    persistent_workers=True if config.training.num_workers > 0 else False,
                    prefetch_factor=prefetch_factor,
                )
        else:
            training_state.train_loader = DataLoader(
                train_ds,
                batch_size=config.training.batch_size,
                shuffle=True,
                num_workers=config.training.num_workers,
                pin_memory=True,
                persistent_workers=True if config.training.num_workers > 0 else False,
                prefetch_factor=prefetch_factor,
            )

        training_state.val_loader = DataLoader(
            val_ds,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=True,
            persistent_workers=True if config.training.num_workers > 0 else False,
            prefetch_factor=prefetch_factor,
        )

        training_state.total_batches = len(training_state.train_loader)

        training_state.add_log("Creating model...")

        # Calculate input size for model
        # This is critical to avoid shape mismatches when changing n_mels or audio_duration
        input_samples = int(config.data.sample_rate * config.data.audio_duration)
        time_steps = input_samples // config.data.hop_length + 1
        
        feature_dim = config.data.n_mels if feature_type == "mel" else config.data.n_mfcc
        
        # Determine input_size based on architecture
        if config.model.architecture == "cd_dnn":
            # CD-DNN expects flattened input (features * time)
            input_size = feature_dim * time_steps
        else:
            # RNNs/TCNs expect feature dimension
            input_size = feature_dim

        # Create model
        training_state.model = create_model(
            architecture=config.model.architecture,
            num_classes=config.model.num_classes,
            pretrained=config.model.pretrained,
            dropout=config.model.dropout,
            input_size=input_size,  # Pass calculated input size
            input_channels=1,       # Always 1 for spectrograms
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
            bidirectional=config.model.bidirectional,
            tcn_num_channels=getattr(config.model, "tcn_num_channels", None),
            tcn_kernel_size=getattr(config.model, "tcn_kernel_size", 3),
        )

        training_state.add_log(f"Model created: {config.model.architecture} (Input Size: {input_size})")

        # Prepare for QAT if enabled
        if config.qat.enabled:
            training_state.add_log(
                f"Preparing model for Quantization Aware Training (Backend: {config.qat.backend})..."
            )
            # We need a dummy input for some QAT preparations (though our current impl doesn't strictly require it)
            # But let's be safe and pass it if we have dimensions
            training_state.model = prepare_model_for_qat(training_state.model, config.qat)
            training_state.add_log("✅ Model prepared for QAT")

        # Run LR Finder if enabled
        optimal_lr = None
        if run_lr_finder:
            training_state.add_log("Running LR Finder (this may take a few minutes)...")
            try:
                optimizer = torch.optim.AdamW(training_state.model.parameters(), lr=config.training.learning_rate)
                criterion = torch.nn.CrossEntropyLoss()
                
                # Wrap model with AudioProcessor if dataset returns raw audio
                model_for_lr = training_state.model
                if getattr(train_ds, "return_raw_audio", False):
                    training_state.add_log("Initializing AudioProcessor for LR Finder...")
                    processor = AudioProcessor(config, cmvn_path=cmvn_path, device="cuda")
                    # Ensure processor is in training mode if needed (though it mostly does stateless ops)
                    processor.train()
                    model_for_lr = torch.nn.Sequential(processor, training_state.model)
                
                # Ensure model is on the correct device
                model_for_lr = model_for_lr.to("cuda")
                
                lr_finder = LRFinder(model_for_lr, optimizer, criterion, device="cuda")

                lrs, losses = lr_finder.range_test(
                    training_state.train_loader,
                    start_lr=1e-6,
                    end_lr=1e-2,
                    num_iter=100,
                )
                optimal_lr = lr_finder.suggest_lr()

                if 1e-5 <= optimal_lr <= 1e-2:
                    config.training.learning_rate = optimal_lr
                    training_state.add_log(f"✅ LR Finder suggested: {optimal_lr:.2e} (applied)")
                else:
                    training_state.add_log(
                        f"⚠️ LR Finder suggested {optimal_lr:.2e} (out of range, keeping {config.training.learning_rate:.2e})"
                    )
            except Exception as e:
                training_state.add_log(f"⚠️ LR Finder failed: {e}")

        # Create checkpoint directory
        checkpoint_dir = paths.CHECKPOINTS
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_manager = CheckpointManager(checkpoint_dir)

        training_state.add_log("Initializing trainer...")

        # Create trainer with EMA if enabled
        TrainerClass = DistillationTrainer if config.distillation.enabled else Trainer

        if config.distillation.enabled:
            training_state.add_log(
                f"Knowledge Distillation enabled (Teacher: {config.distillation.teacher_architecture})"
            )

        training_state.trainer = TrainerClass(
            model=training_state.model,
            train_loader=training_state.train_loader,
            val_loader=training_state.val_loader,
            config=config,
            checkpoint_manager=checkpoint_manager,
            device="cuda",
            use_ema=use_ema,
            ema_decay=ema_decay if use_ema else 0.999,
        )

        # Resume if requested
        start_epoch = 0
        if resume_path:
            training_state.trainer.checkpoint_manager.load_checkpoint(
                resume_path,
                training_state.trainer.model,
                training_state.trainer.optimizer,
                training_state.trainer.device,
            )
            # Load state
            checkpoint = torch.load(resume_path, map_location="cpu")
            start_epoch = checkpoint.get("epoch", 0) + 1
            training_state.current_epoch = start_epoch
            training_state.add_log(f"✅ Resumed training from epoch {start_epoch}")

        if use_ema:
            training_state.add_log(f"✅ EMA enabled (decay: {ema_decay:.4f} → 0.9995)")

        if use_wandb:
            try:
                wandb_callback = WandbCallback(project_name=wandb_project, config=config.to_dict())
                training_state.trainer.add_callback(wandb_callback)
                training_state.add_log("✅ W&B logging enabled")
            except Exception as e:
                training_state.add_log(f"⚠️ W&B initialization failed: {e}. Skipping W&B logging.")

        # Reset history
        training_state.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "val_fpr": [],
            "val_fnr": [],
            "epochs": [],
        }

        # Start training in background thread
        training_state.is_training = True
        training_state.should_stop = False
        training_state.training_thread = threading.Thread(target=training_worker, daemon=True)
        if training_state.training_thread:
            training_state.training_thread.start()

        return (
            "✅ Training started!",
            f"0/{training_state.total_epochs}",
            f"0/{training_state.total_batches}",
            create_loss_plot(),
            create_accuracy_plot(),
            create_metrics_plot(),
        )

    except WakewordException as e:
        error_msg = f"❌ Failed to start training: {str(e)}"
        training_state.add_log(
            f"{error_msg}\n\nActionable suggestion: Please check your configuration and data for the following error: {e}"
        )
        logger.exception("Failed to start training")
        return (error_msg, "0/0", "0/0", None, None, None)
    except Exception as e:
        error_msg = f"❌ Failed to start training: {str(e)}"
        training_state.add_log(error_msg)
        logger.exception("Failed to start training")
        return (error_msg, "0/0", "0/0", None, None, None)


def stop_training() -> str:
    """Stop training"""
    if not training_state.is_training:
        return "⚠️ No training in progress"

    training_state.should_stop = True
    if training_state.trainer:
        training_state.trainer.stop()

    training_state.add_log("Stop requested. Training will stop after current epoch...")

    return "⏹️ Stopping training..."


def get_training_status() -> Tuple:
    """Get current training status for live updates"""
    # Close old matplotlib figures to prevent memory leak
    plt.close("all")

    # Collect logs
    logs = ""
    while not training_state.log_queue.empty():
        try:
            logs += training_state.log_queue.get_nowait()
        except queue.Empty:
            break

    # Status message
    if training_state.is_training:
        status = f"🔄 Training in progress (Epoch {training_state.current_epoch}/{training_state.total_epochs})"
    else:
        status = "✅ Ready to train"

    # Calculate ETA (simple estimation)
    if training_state.is_training and training_state.current_epoch > 0:
        # Rough estimate based on current progress
        epochs_remaining = training_state.total_epochs - training_state.current_epoch
        # Assume similar time per epoch
        training_state.eta_seconds = epochs_remaining * 60  # Placeholder
    else:
        training_state.eta_seconds = 0

    # GPU utilization
    try:
        validator = get_cuda_validator()
        gpu_util = validator.get_memory_info()
        gpu_percent = gpu_util["allocated_gb"] / gpu_util["total_gb"] * 100
    except:
        gpu_percent = 0.0

    return (
        status,
        f"{training_state.current_epoch}/{training_state.total_epochs}",
        f"{training_state.current_batch}/{training_state.total_batches}",
        round(training_state.current_train_loss, 4),
        round(training_state.current_val_loss, 4),
        round(training_state.current_train_acc * 100, 2),
        round(training_state.current_val_acc * 100, 2),
        round(training_state.current_fpr * 100, 2),
        round(training_state.current_fnr * 100, 2),
        round(training_state.current_speed, 1),
        round(gpu_percent, 1),
        format_time(training_state.eta_seconds),
        logs,
        create_loss_plot(),
        create_accuracy_plot(),
        create_metrics_plot(),
        str(training_state.best_epoch),
        round(training_state.best_val_loss, 4),
        round(training_state.best_val_acc * 100, 2),
        training_state.best_model_path,
    )


def hpo_worker(config: WakewordConfig, n_trials: int, study_name: str) -> None:
    """Background worker for HPO"""
    try:
        training_state.add_log(f"Starting HPO study '{study_name}'...")

        # Use centralized paths
        splits_dir = paths.SPLITS

        if not splits_dir.exists() or not (splits_dir / "train.json").exists():
            training_state.add_log("❌ Dataset splits not found.")
            return

        train_ds, val_ds, _ = load_dataset_splits(
            data_root=paths.DATA,
            sample_rate=config.data.sample_rate,
            audio_duration=config.data.audio_duration,
            augment_train=True,
            augmentation_config=config.augmentation.to_dict(),
            device="cuda",
            feature_type=config.data.feature_type,
            n_mels=config.data.n_mels,
            n_mfcc=config.data.n_mfcc,
            n_fft=config.data.n_fft,
            hop_length=config.data.hop_length,
            use_precomputed_features_for_training=config.data.use_precomputed_features_for_training,
            npy_cache_features=config.data.npy_cache_features,
            fallback_to_audio=True,  # Force True for HPO robustness
            cmvn_path=paths.DATA / "cmvn_stats.json",
            apply_cmvn=True,
            return_raw_audio=True,  # NEW: Use GPU pipeline for HPO
        )

        # Optimize config for GPU pipeline
        hpo_config = config.copy()
        hpo_config.training.num_workers = min(hpo_config.training.num_workers, 16)
        hpo_config.training.pin_memory = True

        train_loader = DataLoader(
            train_ds,
            batch_size=hpo_config.training.batch_size,
            shuffle=True,
            num_workers=hpo_config.training.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=hpo_config.training.batch_size,
            shuffle=False,
            num_workers=hpo_config.training.num_workers,
            pin_memory=True,
        )

        # Run HPO with logging callback
        study = run_hpo(hpo_config, train_loader, val_loader, n_trials, study_name, log_callback=training_state.add_log)

        training_state.add_log(f"✅ HPO study '{study_name}' finished successfully.")

    except Exception as e:
        training_state.add_log(f"❌ HPO failed: {e}")
        logger.exception("HPO failed")
    finally:
        training_state.is_training = False


def start_hpo(state: gr.State, n_trials: int, n_jobs: int, study_name: str, param_groups: List[str]) -> Tuple[str, pd.DataFrame]:
    """Start HPO study in background"""
    if training_state.is_training:
        return "⚠️ Training already in progress", None

    if not param_groups:
        return "⚠️ Please select at least one parameter group to optimize", None

    training_state.reset()
    training_state.is_training = True
    training_state.should_stop = False

    def hpo_thread_func() -> None:
        try:
            config = state["config"]
            train_loader = training_state.train_loader
            val_loader = training_state.val_loader

            # Load data if needed
            if train_loader is None or val_loader is None:
                training_state.add_log("Datasets not loaded. Loading for HPO...")

                # Use centralized paths
                splits_dir = paths.SPLITS
                if not splits_dir.exists() or not (splits_dir / "train.json").exists():
                    training_state.add_log("❌ Dataset splits not found. Please run Panel 1 first.")
                    return

                # Normalize feature type
                feature_type = "mel" if config.data.feature_type == "mel_spectrogram" else config.data.feature_type

                # Check CMVN stats (basic check, full check is in start_training)
                cmvn_path = paths.CMVN_STATS
                if not cmvn_path.exists():
                    training_state.add_log("⚠️ CMVN stats missing. Computing...")
                    temp_train_ds, _, _ = load_dataset_splits(
                        data_root=paths.DATA,
                        device="cuda",
                        feature_type=feature_type,
                        n_mels=config.data.n_mels,
                        n_mfcc=config.data.n_mfcc,
                        n_fft=config.data.n_fft,
                        hop_length=config.data.hop_length,
                        use_precomputed_features_for_training=config.data.use_precomputed_features_for_training,
                        fallback_to_audio=True,
                        apply_cmvn=False,
                    )
                    compute_cmvn_from_dataset(temp_train_ds, cmvn_path, max_samples=1000)

                train_ds, val_ds, _ = load_dataset_splits(
                    data_root=paths.DATA,
                    sample_rate=config.data.sample_rate,
                    audio_duration=config.data.audio_duration,
                    augment_train=True,
                    augmentation_config=config.augmentation.to_dict(),
                    device="cuda",
                    feature_type=feature_type,
                    n_mels=config.data.n_mels,
                    n_mfcc=config.data.n_mfcc,
                    n_fft=config.data.n_fft,
                    hop_length=config.data.hop_length,
                    use_precomputed_features_for_training=config.data.use_precomputed_features_for_training,
                    npy_cache_features=config.data.npy_cache_features,
                    fallback_to_audio=True,
                    cmvn_path=cmvn_path,
                    apply_cmvn=True,
                    return_raw_audio=True,
                )

                # Create loaders
                hpo_config = config.copy()
                hpo_config.training.num_workers = min(hpo_config.training.num_workers, 16)
                hpo_config.training.pin_memory = True

                train_loader = DataLoader(
                    train_ds,
                    batch_size=hpo_config.training.batch_size,
                    shuffle=True,
                    num_workers=hpo_config.training.num_workers,
                    pin_memory=True,
                )
                val_loader = DataLoader(
                    val_ds,
                    batch_size=hpo_config.training.batch_size,
                    shuffle=False,
                    num_workers=hpo_config.training.num_workers,
                    pin_memory=True,
                )

                training_state.add_log(f"✅ Loaded {len(train_ds)} training samples for HPO")

            result = run_hpo(
                config=state["config"],
                train_loader=train_loader,
                val_loader=val_loader,
                n_trials=int(n_trials),
                study_name=study_name,
                param_groups=param_groups,
                log_callback=training_state.add_log,
                n_jobs=int(n_jobs),
            )
            training_state.add_log("✅ HPO Study Complete!")
            # Store best params in state for "Apply" feature
            state["best_hpo_params"] = result.best_params

        except Exception as e:
            logger.exception("HPO failed")
            training_state.add_log(f"❌ HPO failed: {str(e)}")
        finally:
            training_state.is_training = False

    training_state.training_thread = threading.Thread(target=hpo_thread_func)
    if training_state.training_thread:
        training_state.training_thread.start()

    return f"🚀 HPO study '{study_name}' started. Optimizing: {', '.join(param_groups)} (Jobs: {n_jobs})", None



def apply_best_params(state: gr.State) -> str:
    """Apply best HPO params to current config"""
    if "best_hpo_params" not in state or not state["best_hpo_params"]:
        return "⚠️ No HPO results found to apply. Run HPO first."

    best_params = state["best_hpo_params"]
    config = state["config"]

    # Map flat params to config structure
    # Training
    if "learning_rate" in best_params:
        config.training.learning_rate = best_params["learning_rate"]
    if "batch_size" in best_params:
        config.training.batch_size = best_params["batch_size"]
    if "weight_decay" in best_params:
        config.optimizer.weight_decay = best_params["weight_decay"]
    if "optimizer" in best_params:
        config.optimizer.optimizer = best_params["optimizer"]

    # Model
    if "dropout" in best_params:
        config.model.dropout = best_params["dropout"]
    if "hidden_size" in best_params:
        config.model.hidden_size = best_params["hidden_size"]

    # Augmentation
    if "background_noise_prob" in best_params:
        config.augmentation.background_noise_prob = best_params["background_noise_prob"]
    if "rir_prob" in best_params:
        config.augmentation.rir_prob = best_params["rir_prob"]
    if "time_stretch_min" in best_params:
        config.augmentation.time_stretch_min = best_params["time_stretch_min"]
    if "time_stretch_max" in best_params:
        config.augmentation.time_stretch_max = best_params["time_stretch_max"]
    if "freq_mask_param" in best_params:
        config.augmentation.freq_mask_param = best_params["freq_mask_param"]
    if "time_mask_param" in best_params:
        config.augmentation.time_mask_param = best_params["time_mask_param"]

    # Data
    if "n_mels" in best_params:
        config.data.n_mels = best_params["n_mels"]

    # Loss
    if "loss_function" in best_params:
        config.loss.loss_function = best_params["loss_function"]
    if "focal_gamma" in best_params:
        config.loss.focal_gamma = best_params["focal_gamma"]

    return f"✅ Applied best parameters: {best_params}"


def save_best_profile(state: gr.State) -> str:
    """Save best HPO params as a new user profile"""
    if "best_hpo_params" not in state or not state["best_hpo_params"]:
        return "⚠️ No HPO results found to save."

    # Apply params first to ensure config is up to date
    apply_best_params(state)

    # Create filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    profile_name = f"user_profile_{timestamp}"
    filename = f"{profile_name}.yaml"
    save_path = paths.CONFIGS / filename

    # Update config metadata
    state["config"].config_name = profile_name
    state["config"].description = f"User generated profile from HPO results on {timestamp} (User)"

    # Save
    try:
        state["config"].save(save_path)
        return f"💾 Saved profile to {filename}"
    except Exception as e:
        return f"❌ Failed to save profile: {e}"


def list_checkpoints() -> List[str]:
    """List available checkpoints"""
    checkpoints_dir = paths.CHECKPOINTS
    if not checkpoints_dir.exists():
        return []
    return [f.name for f in checkpoints_dir.glob("*.pt")]


def load_wandb_key() -> str:
    """Load WandB API key from file"""
    key_file = Path(".wandb_key")
    if key_file.exists():
        try:
            return key_file.read_text().strip()
        except Exception:
            return ""
    return ""


def save_wandb_key(key: str) -> None:
    """Save WandB API key to file"""
    if not key:
        return
    try:
        Path(".wandb_key").write_text(key.strip())
    except Exception as e:
        print(f"Failed to save WandB key: {e}")


def calculate_dataset_ratios() -> Tuple[str, float, float, float]:
    """Calculate optimal ratios from train.json"""
    try:
        train_manifest = paths.SPLITS / "train.json"
        if not train_manifest.exists():
            return "❌ train.json not found", 1, 1, 1

        import json

        with open(train_manifest, "r") as f:
            data = json.load(f)

        # Count classes
        counts = {"positive": 0, "negative": 0, "hard_negative": 0}

        # Handle both list (legacy) and dict (new) formats
        items = data
        if isinstance(data, dict):
            items = data.get("files", [])

        for item in items:
            cat = item.get("category", "negative")
            if cat in counts:
                counts[cat] += 1

        if sum(counts.values()) == 0:
            return "❌ Dataset empty", 1, 1, 1

        # Find min non-zero count to normalize
        min_count = min([c for c in counts.values() if c > 0], default=1)

        # Calculate ratios (rounded to nearest integer for cleaner UI)
        r_pos = max(1, round(counts["positive"] / min_count))
        r_neg = max(1, round(counts["negative"] / min_count))
        r_hard = max(1, round(counts["hard_negative"] / min_count))

        msg = (
            f"✅ Calculated from {sum(counts.values())} samples:\n"
            f"Pos: {counts['positive']} ({r_pos}x)\n"
            f"Neg: {counts['negative']} ({r_neg}x)\n"
            f"Hard: {counts['hard_negative']} ({r_hard}x)"
        )
        return msg, r_pos, r_neg, r_hard

    except Exception as e:
        return f"❌ Error: {str(e)}", 1, 1, 1


def create_training_panel(state: gr.State) -> gr.Blocks:
    """
    Create Panel 3: Model Training

    Args:
        state: Global state dictionary

    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks() as panel:
        gr.Markdown("# 🚀 Model Training")
        gr.Markdown("Train your wakeword model with real-time monitoring and GPU acceleration.")

        # Advanced Features Section
        with gr.Accordion("⚙️ Advanced Training Features", open=False):
            gr.Markdown("### Production-Ready Features")
            gr.Markdown("Enable advanced features for improved model quality and training efficiency.")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🔧 CMVN Normalization")
                    use_cmvn = gr.Checkbox(
                        label="Enable CMVN (Cepstral Mean Variance Normalization)",
                        value=True,
                        info="Corpus-level feature normalization for consistent features (+2-4% accuracy)",
                    )

                with gr.Column():
                    gr.Markdown("#### 📊 EMA (Exponential Moving Average)")
                    use_ema = gr.Checkbox(
                        label="Enable EMA",
                        value=True,
                        info="Shadow model weights for stable inference (+1-2% validation accuracy)",
                    )
                    ema_decay = gr.Slider(
                        minimum=0.99,
                        maximum=0.9999,
                        value=0.999,
                        step=0.0001,
                        label="EMA Decay",
                        info="Initial decay rate (auto-adjusts to 0.9995 in final epochs)",
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### ⚖️ Balanced Batch Sampling")
                    use_balanced_sampler = gr.Checkbox(
                        label="Enable Balanced Sampler",
                        value=True,
                        info="Control pos:neg:hard_neg ratios in batches",
                    )
                    with gr.Row():
                        sampler_ratio_pos = gr.Number(
                            label="Positive",
                            value=1,
                            precision=0,
                            minimum=1,
                            info="Ratio of positive samples",
                        )
                        sampler_ratio_neg = gr.Number(
                            label="Negative",
                            value=1,
                            precision=0,
                            minimum=1,
                            info="Ratio of negative samples",
                        )
                        sampler_ratio_hard = gr.Number(
                            label="Hard Negative",
                            value=1,
                            precision=0,
                            minimum=0,
                            info="Ratio of hard negatives",
                        )

                    calc_ratios_btn = gr.Button("🧮 Auto-Calculate Ratios", size="sm")
                    ratio_status = gr.Textbox(label="Ratio Status", value="", interactive=False, lines=2)

                with gr.Column():
                    gr.Markdown("#### 🔄 Resume Training")

                    with gr.Row():
                        checkpoint_dropdown = gr.Dropdown(
                            label="Select Checkpoint",
                            choices=list_checkpoints(),
                            value=None,
                            interactive=True,
                            info="Select a .pt file to resume from",
                        )
                        refresh_ckpt_btn = gr.Button("🔄", size="sm", scale=0)

                    resume_training = gr.Checkbox(
                        label="Resume from selected",
                        value=False,
                        info="Load weights and optimizer state from checkpoint",
                    )

                with gr.Column():
                    gr.Markdown("#### 🔍 Learning Rate Finder")
                    run_lr_finder = gr.Checkbox(
                        label="Run LR Finder",
                        value=False,
                        info="Automatically discover optimal learning rate (-10-15% training time)",
                    )
                    gr.Markdown("*Note: LR Finder runs before training starts and may take a few minutes*")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📈 Experiment Tracking (Weights & Biases)")
                    use_wandb = gr.Checkbox(
                        label="Enable W&B Logging",
                        value=False,
                        info="Log metrics, parameters, and model artifacts to Weights & Biases.",
                    )
                    wandb_project = gr.Textbox(
                        label="W&B Project Name",
                        value="wakeword-training",
                        info="The name of the project in Weights & Biases.",
                    )
                    wandb_api_key = gr.Textbox(
                        label="W&B API Key",
                        value=load_wandb_key(),  # Load saved key
                        placeholder="Paste your API key here",
                        type="password",
                        info="Found in W&B Settings > API Keys. Leave empty if already logged in via CLI.",
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📉 Loss Configuration")
                    loss_function = gr.Dropdown(
                        label="Loss Function",
                        choices=["cross_entropy", "focal_loss", "triplet_loss"],
                        value="cross_entropy",
                        info="Objective function for optimization",
                    )
                    label_smoothing = gr.Slider(
                        minimum=0.0,
                        maximum=0.5,
                        value=0.05,
                        step=0.01,
                        label="Label Smoothing",
                    )
                with gr.Column():
                    gr.Markdown("#### 🎯 Focal Loss Params")
                    focal_gamma = gr.Slider(
                        minimum=0.0,
                        maximum=5.0,
                        value=2.0,
                        step=0.1,
                        label="Focal Gamma",
                        info="Focus on hard examples (Higher = more focus)",
                    )
                    focal_alpha = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.25,
                        step=0.05,
                        label="Focal Alpha",
                        info="Balance for positive class",
                    )
                with gr.Column():
                    gr.Markdown("#### 🧱 Mining")
                    hard_negative_weight = gr.Slider(
                        minimum=1.0,
                        maximum=5.0,
                        value=1.5,
                        step=0.1,
                        label="Hard Negative Weight",
                        info="Penalty multiplier for hard negatives",
                    )

        gr.Markdown("---")

        with gr.Tabs():
            with gr.TabItem("🧠 Training"):
                with gr.Row():
                    start_training_btn = gr.Button("▶️ Start Training", variant="primary", scale=2)
                    stop_training_btn = gr.Button("⏹️ Stop Training", variant="stop", scale=1)
            with gr.TabItem("🔬 Hyperparameter Optimization"):
                with gr.Row():
                    hpo_status = gr.Textbox(
                        label="HPO Status",
                        value="Ready to start HPO",
                        interactive=False,
                    )
                with gr.Row():
                    n_trials = gr.Slider(
                        minimum=10,
                        maximum=200,
                        value=50,
                        step=10,
                        label="Number of Trials",
                    )
                    n_jobs = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=1,
                        step=1,
                        label="Parallel Jobs",
                        info="Number of parallel trials (increases RAM usage)",
                    )
                    study_name = gr.Textbox(label="Study Name", value="wakeword-hpo")
                with gr.Row():
                    start_hpo_btn = gr.Button("🚀 Start HPO Study", variant="primary")

                with gr.Row():
                    hpo_param_groups = gr.CheckboxGroup(
                        label="Parameter Groups to Optimize",
                        choices=["Training", "Model", "Augmentation", "Data", "Loss"],
                        value=["Training", "Model"],
                        info="Select which parameters to include in the search space",
                    )

                with gr.Row():
                    apply_params_btn = gr.Button("✅ Apply Best Params", size="sm")
                    save_profile_btn = gr.Button("💾 Save as Profile", size="sm")
                    hpo_action_status = gr.Textbox(label="Action Status", interactive=False)

                with gr.Row():
                    hpo_results = gr.DataFrame(headers=["Trial", "Value", "Params"])

        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Training Status")
                training_status = gr.Textbox(label="Status", value="Ready to train", lines=2, interactive=False)

                gr.Markdown("### Current Progress")
                current_epoch = gr.Textbox(label="Epoch", value="0/0", interactive=False)
                current_batch = gr.Textbox(label="Batch", value="0/0", interactive=False)

                gr.Markdown("### Current Metrics")
                with gr.Row():
                    train_loss = gr.Number(label="Train Loss", value=0.0, interactive=False)
                    val_loss = gr.Number(label="Val Loss", value=0.0, interactive=False)

                with gr.Row():
                    train_acc = gr.Number(label="Train Acc (%)", value=0.0, interactive=False)
                    val_acc = gr.Number(label="Val Acc (%)", value=0.0, interactive=False)

                with gr.Row():
                    fpr = gr.Number(label="FPR (%)", value=0.0, interactive=False)
                    fnr = gr.Number(label="FNR (%)", value=0.0, interactive=False)

                with gr.Row():
                    speed = gr.Number(label="Speed (samples/sec)", value=0.0, interactive=False)
                    gpu_util = gr.Number(label="GPU Util (%)", value=0.0, interactive=False)

                eta = gr.Textbox(label="ETA", value="--:--:--", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("### Training Curves")

                # Loss plot
                loss_plot = gr.Plot(label="Loss Curves", value=create_loss_plot())

                # Accuracy plot
                accuracy_plot = gr.Plot(label="Accuracy Curves", value=create_accuracy_plot())

                # Metrics plot
                metrics_plot = gr.Plot(
                    label="Validation Metrics (FPR, FNR, F1)",
                    value=create_metrics_plot(),
                )

        gr.Markdown("---")

        with gr.Row():
            gr.Markdown("### Training Log")

        with gr.Row():
            training_log = gr.Textbox(
                label="Console Output",
                lines=8,
                value="Waiting to start training...\n",
                interactive=False,
                max_lines=100,
                autoscroll=True,
            )

        gr.Markdown("---")

        with gr.Row():
            gr.Markdown("### Best Model Info")

        with gr.Row():
            with gr.Column():
                best_epoch = gr.Textbox(label="Best Epoch", value="--", interactive=False)
                best_val_loss = gr.Number(label="Best Val Loss", value=0.0, interactive=False)
            with gr.Column():
                best_val_acc = gr.Number(label="Best Val Acc (%)", value=0.0, interactive=False)
                model_path = gr.Textbox(
                    label="Checkpoint Path",
                    value="No model saved yet",
                    interactive=False,
                )

        # Event handlers
        # Wrapper for start training to handle resume logic
        def start_training_wrapper(*args: Any) -> Any:
            # Last 2 args are resume_training (bool) and checkpoint_dropdown (str)
            resume_checked = args[-2]
            ckpt_path = args[-1]

            # Pass everything else + formatted resume path
            actual_args = list(args[:-2])
            actual_args.append(ckpt_path if resume_checked else None)

            return start_training(*actual_args)

        start_training_btn.click(
            fn=start_training_wrapper,
            inputs=[
                state,
                use_cmvn,
                use_ema,
                ema_decay,
                use_balanced_sampler,
                sampler_ratio_pos,
                sampler_ratio_neg,
                sampler_ratio_hard,
                run_lr_finder,
                use_wandb,
                wandb_project,
                loss_function,
                label_smoothing,
                focal_gamma,
                focal_alpha,
                hard_negative_weight,
                wandb_api_key,
                resume_training,
                checkpoint_dropdown,
            ],
            outputs=[
                training_status,
                current_epoch,
                current_batch,
                loss_plot,
                accuracy_plot,
                metrics_plot,
            ],
        )

        # Auto-Calculate Ratios Handler
        calc_ratios_btn.click(
            fn=calculate_dataset_ratios,
            inputs=[],
            outputs=[ratio_status, sampler_ratio_pos, sampler_ratio_neg, sampler_ratio_hard],
        )

        # Refresh Checkpoints Handler
        def refresh_checkpoints() -> gr.Dropdown:
            return gr.Dropdown(choices=list_checkpoints())

        refresh_ckpt_btn.click(fn=refresh_checkpoints, inputs=[], outputs=[checkpoint_dropdown])

        stop_training_btn.click(fn=stop_training, outputs=[training_status])

        start_hpo_btn.click(
            fn=start_hpo,
            inputs=[state, n_trials, n_jobs, study_name, hpo_param_groups],
            outputs=[hpo_status, hpo_results],
        )

        apply_params_btn.click(fn=apply_best_params, inputs=[state], outputs=[hpo_action_status])

        save_profile_btn.click(fn=save_best_profile, inputs=[state], outputs=[hpo_action_status])

        # Auto-refresh for live updates
        status_refresh = gr.Timer(value=2.0, active=True)  # Update every 2 seconds

        status_refresh.tick(
            fn=get_training_status,
            outputs=[
                training_status,
                current_epoch,
                current_batch,
                train_loss,
                val_loss,
                train_acc,
                val_acc,
                fpr,
                fnr,
                speed,
                gpu_util,
                eta,
                training_log,
                loss_plot,
                accuracy_plot,
                metrics_plot,
                best_epoch,
                best_val_loss,
                best_val_acc,
                model_path,
            ],
        )

    # Expose component for app.py
    panel.wandb_api_key = wandb_api_key

    return panel


if __name__ == "__main__":
    # Test the panel
    state = gr.State(value={})
    demo = create_training_panel(state)
    demo.launch()
