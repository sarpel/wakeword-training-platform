import shutil
import tempfile
from pathlib import Path
from typing import Callable, List, Optional

import optuna
import structlog
from torch.utils.data import DataLoader

from src.config.defaults import WakewordConfig
from src.models.architectures import create_model
from src.training.checkpoint_manager import CheckpointManager
from src.training.metrics import MetricResults
from src.training.trainer import Trainer

logger = structlog.get_logger(__name__)


class OptunaPruningCallback:
    """Callback to prune unpromising trials."""

    def __init__(
        self,
        trial: optuna.trial.Trial,
        monitor: str = "f1_score",
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        self.trial = trial
        self.monitor = monitor
        self.log_callback = log_callback

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float, val_metrics: MetricResults) -> None:
        # Report current score to Optuna
        # val_metrics is a MetricResults object with attributes like f1_score
        current_score = getattr(val_metrics, self.monitor, 0.0)

        # Report intermediate objective value
        self.trial.report(current_score, epoch)

        # Handle pruning based on the reported value
        if self.trial.should_prune():
            message = f"Trial pruned at epoch {epoch} with {self.monitor}={current_score:.4f}"
            logger.info(message)
            if self.log_callback:
                self.log_callback(f"✂️ {message}")
            raise optuna.TrialPruned(message)


class Objective:
    """Optuna objective for hyperparameter optimization."""

    def __init__(
        self,
        config: WakewordConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        param_groups: Optional[List[str]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.param_groups = param_groups or ["Training", "Model", "Augmentation"]
        self.best_f1 = -1.0
        self.log_callback = log_callback

    def _log(self, message: str) -> None:
        """Log to both structlog and callback if available"""
        logger.info(message)
        if self.log_callback:
            self.log_callback(message)

    def __call__(self, trial: optuna.trial.Trial) -> float:
        """Run a single training trial with a set of hyperparameters."""
        # Create a copy of the config to avoid side effects
        trial_config = self.config.copy()

        # --- 1. Expand Search Space based on param_groups ---

        # Group: Training
        if "Training" in self.param_groups:
            trial_config.training.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            trial_config.optimizer.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
            trial_config.optimizer.optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw"])
            # Batch size (requires recreating DataLoaders)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            trial_config.training.batch_size = batch_size

        # Group: Model
        if "Model" in self.param_groups:
            trial_config.model.dropout = trial.suggest_float("dropout", 0.1, 0.5)
            # Only suggest hidden_size if architecture supports it (RNNs)
            if trial_config.model.architecture in ["lstm", "gru"]:
                trial_config.model.hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])

        # Group: Augmentation
        if "Augmentation" in self.param_groups:
            trial_config.augmentation.background_noise_prob = trial.suggest_float("background_noise_prob", 0.1, 0.9)
            trial_config.augmentation.rir_prob = trial.suggest_float("rir_prob", 0.1, 0.8)
            trial_config.augmentation.time_stretch_min = trial.suggest_float("time_stretch_min", 0.8, 0.95)
            trial_config.augmentation.time_stretch_max = trial.suggest_float("time_stretch_max", 1.05, 1.2)
            # SpecAugment parameters
            trial_config.augmentation.freq_mask_param = trial.suggest_int("freq_mask_param", 10, 40)
            trial_config.augmentation.time_mask_param = trial.suggest_int("time_mask_param", 20, 60)

        # Group: Data
        if "Data" in self.param_groups:
            # Only optimize n_mels if we are NOT using precomputed features
            # If using precomputed features, we must stick to the dimension of the files
            if not self.config.data.use_precomputed_features_for_training:
                # Be careful with n_mels as it changes input size
                n_mels = trial.suggest_categorical("n_mels", [40, 64, 80])
                trial_config.data.n_mels = n_mels
            else:
                self._log(f"Skipping n_mels optimization (using precomputed features: {self.config.data.n_mels})")

        # Group: Loss
        if "Loss" in self.param_groups:
            trial_config.loss.loss_function = trial.suggest_categorical(
                "loss_function", ["cross_entropy", "focal_loss"]
            )
            if trial_config.loss.loss_function == "focal_loss":
                trial_config.loss.focal_gamma = trial.suggest_float("focal_gamma", 1.0, 4.0)

        # Ensure batch_size is set if not optimized
        batch_size = trial_config.training.batch_size

        # --- 2. Epoch Reduction (Fidelity) ---
        # Run for fewer epochs during HPO to save time
        HPO_EPOCHS = 20
        trial_config.training.epochs = HPO_EPOCHS

        # Early stopping relative to HPO epochs
        trial_config.training.early_stopping_patience = 5

        self._log(f"Trial {trial.number} started. Params: {trial.params}")

        # Re-create DataLoaders with new batch size
        train_loader_trial = DataLoader(
            self.train_loader.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
            persistent_workers=True if self.config.training.num_workers > 0 else False,
        )
        val_loader_trial = DataLoader(
            self.val_loader.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
            persistent_workers=True if self.config.training.num_workers > 0 else False,
        )

        # Create a temporary directory for checkpoints for this trial
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            checkpoint_manager = CheckpointManager(checkpoint_dir)

            # Determine input size based on feature type
            feature_dim = 64  # Default
            if trial_config.data.feature_type == "mfcc":
                feature_dim = trial_config.data.n_mfcc
            else:
                feature_dim = trial_config.data.n_mels

            # Calculate time steps for CD-DNN
            input_samples = int(trial_config.data.sample_rate * trial_config.data.audio_duration)
            time_steps = input_samples // trial_config.data.hop_length + 1

            if trial_config.model.architecture == "cd_dnn":
                input_size = feature_dim * time_steps
            else:
                input_size = feature_dim

            # Create a new model for this trial
            model = create_model(
                architecture=trial_config.model.architecture,
                num_classes=trial_config.model.num_classes,
                dropout=trial_config.model.dropout,
                input_channels=1,
                input_size=input_size,  # Pass correct input size
            )

            # Create a new trainer for this trial
            trainer = Trainer(
                model=model,
                train_loader=train_loader_trial,
                val_loader=val_loader_trial,
                config=trial_config,
                checkpoint_manager=checkpoint_manager,
                device="cuda",
            )

            # --- 3. Early Pruning ---
            # Add pruning callback
            pruning_callback = OptunaPruningCallback(trial, monitor="f1_score", log_callback=self.log_callback)
            trainer.add_callback(pruning_callback)

            try:
                results = trainer.train()

                best_f1 = results["best_val_f1"]
                best_fpr = results["best_val_fpr"]

                # --- 4. FPR Constraint ---
                # Penalize models with high False Positive Rate (> 5%)
                if best_fpr > 0.05:
                    self._log(f"Trial {trial.number} penalized due to high FPR: {best_fpr:.4f}")
                    return 0.0

                metric = best_f1

                # Save if best so far
                if metric > self.best_f1:
                    self.best_f1 = metric
                    # Create models dir if not exists
                    save_path = Path("models/hpo_best_model.pt")
                    save_path.parent.mkdir(parents=True, exist_ok=True)

                    source_path = checkpoint_dir / "best_model.pt"
                    if source_path.exists():
                        shutil.copy(source_path, save_path)
                        self._log(f"🏆 New best HPO model saved to {save_path} (F1: {metric:.4f})")

                self._log(f"Trial {trial.number} finished. F1: {metric:.4f}")

            except optuna.TrialPruned:
                raise
            except Exception as e:
                self._log(f"❌ Trial {trial.number} failed: {e}")
                metric = 0.0

            return float(metric)


def run_hpo(
    config: WakewordConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_trials: int = 50,
    study_name: str = "wakeword-hpo",
    param_groups: Optional[List[str]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> optuna.study.Study:
    """Run hyperparameter optimization using Optuna."""
    objective = Objective(config, train_loader, val_loader, param_groups, log_callback=log_callback)

    # Use MedianPruner to stop unpromising trials early
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    study = optuna.create_study(direction="maximize", study_name=study_name, pruner=pruner, load_if_exists=True)

    if log_callback:
        log_callback(f"Starting HPO study '{study_name}' with {n_trials} trials.")
    logger.info(f"Starting HPO study '{study_name}' with {n_trials} trials.")

    study.optimize(objective, n_trials=n_trials)

    if log_callback:
        log_callback(f"✅ HPO Complete. Best trial: {study.best_trial.value}")
        log_callback(f"Best params: {study.best_trial.params}")

    logger.info(f"Best trial: {study.best_trial.value}")
    logger.info(f"Best params: {study.best_trial.params}")

    return study
