import optuna
from typing import Dict, Any
from pathlib import Path
import tempfile
import shutil
import torch
from torch.utils.data import DataLoader
from src.training.trainer import Trainer
from src.config.defaults import WakewordConfig
from src.models.architectures import create_model
from src.training.checkpoint_manager import CheckpointManager
import logging
import structlog

logger = structlog.get_logger(__name__)

class OptunaPruningCallback:
    """Callback to prune unpromising trials."""
    
    def __init__(self, trial: optuna.trial.Trial, monitor: str = "f1_score"):
        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float, val_metrics):
        # Report current score to Optuna
        # val_metrics is a MetricResults object with attributes like f1_score
        current_score = getattr(val_metrics, self.monitor, 0.0)
        
        # Report intermediate objective value
        self.trial.report(current_score, epoch)

        # Handle pruning based on the reported value
        if self.trial.should_prune():
            message = f"Trial pruned at epoch {epoch} with {self.monitor}={current_score:.4f}"
            logger.info(message)
            raise optuna.TrialPruned(message)

class Objective:
    """Optuna objective for hyperparameter optimization."""

    def __init__(self, config: WakewordConfig, train_loader: DataLoader, val_loader: DataLoader):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

    def __call__(self, trial: optuna.trial.Trial) -> float:
        """Run a single training trial with a set of hyperparameters."""
        # Create a copy of the config to avoid side effects
        trial_config = self.config.copy()

        # --- 1. Expand Search Space ---
        # Optimizer hyperparameters
        trial_config.training.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        trial_config.optimizer.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        
        # Model hyperparameters
        trial_config.model.dropout = trial.suggest_float("dropout", 0.1, 0.5)
        
        # Augmentation hyperparameters
        trial_config.augmentation.background_noise_prob = trial.suggest_float("background_noise_prob", 0.1, 0.7)
        trial_config.augmentation.time_stretch_min = trial.suggest_float("time_stretch_min", 0.8, 0.95)
        trial_config.augmentation.time_stretch_max = trial.suggest_float("time_stretch_max", 1.05, 1.2)

        # Batch size (requires recreating DataLoaders)
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        trial_config.training.batch_size = batch_size

        # --- 2. Epoch Reduction (Fidelity) ---
        # Run for fewer epochs during HPO to save time
        HPO_EPOCHS = 20
        trial_config.training.epochs = HPO_EPOCHS
        
        # Early stopping relative to HPO epochs
        trial_config.training.early_stopping_patience = 5 

        # Re-create DataLoaders with new batch size
        train_loader_trial = DataLoader(
            self.train_loader.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory
        )
        val_loader_trial = DataLoader(
            self.val_loader.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory
        )

        # Create a temporary directory for checkpoints for this trial
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            checkpoint_manager = CheckpointManager(checkpoint_dir)
            
            # Determine input size based on feature type
            input_size = 64  # Default
            if trial_config.data.feature_type == "mfcc":
                input_size = trial_config.data.n_mfcc
            else:
                input_size = trial_config.data.n_mels

            # Create a new model for this trial
            model = create_model(
                architecture=trial_config.model.architecture,
                num_classes=trial_config.model.num_classes,
                dropout=trial_config.model.dropout,
                input_channels=1,
                input_size=input_size  # Pass correct input size
            )

            # Create a new trainer for this trial
            trainer = Trainer(
                model=model,
                train_loader=train_loader_trial,
                val_loader=val_loader_trial,
                config=trial_config,
                checkpoint_manager=checkpoint_manager,
                device='cuda'
            )
            
            # --- 3. Early Pruning ---
            # Add pruning callback
            pruning_callback = OptunaPruningCallback(trial, monitor="f1_score")
            trainer.add_callback(pruning_callback)

            try:
                results = trainer.train()
                
                best_f1 = results["best_val_f1"]
                best_fpr = results["best_val_fpr"]
                
                # --- 4. FPR Constraint ---
                # Penalize models with high False Positive Rate (> 5%)
                if best_fpr > 0.05:
                    logger.info(f"Trial penalized due to high FPR: {best_fpr:.4f}")
                    return 0.0
                
                metric = best_f1
                
            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.error(f"Trial failed: {e}")
                metric = 0.0
            
            return metric

def run_hpo(
    config: WakewordConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_trials: int = 50,
    study_name: str = "wakeword-hpo",
) -> optuna.study.Study:
    """Run hyperparameter optimization using Optuna."""
    objective = Objective(config, train_loader, val_loader)
    
    # Use MedianPruner to stop unpromising trials early
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    
    study = optuna.create_study(
        direction="maximize", 
        study_name=study_name,
        pruner=pruner,
        load_if_exists=True
    )
    
    logger.info(f"Starting HPO study '{study_name}' with {n_trials} trials.")
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Best trial: {study.best_trial.value}")
    logger.info(f"Best params: {study.best_trial.params}")

    return study
