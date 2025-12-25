"""
Training Module for Wakeword Detection
"""

from src.training.advanced_metrics import calculate_eer, calculate_fah, calculate_pauc
from src.training.checkpoint_manager import CheckpointManager
from src.training.distillation_trainer import DistillationTrainer
from src.training.ema import EMA
from src.training.hpo import run_hpo
from src.training.hpo_results import HPOResult
from src.training.lr_finder import LRFinder
from src.training.metrics import MetricResults
from src.training.qat_utils import prepare_model_for_qat
from src.training.trainer import Trainer
from src.training.wandb_callback import WandbCallback

__all__ = [
    "Trainer",
    "DistillationTrainer",
    "CheckpointManager",
    "EMA",
    "LRFinder",
    "run_hpo",
    "HPOResult",
    "MetricResults",
    "WandbCallback",
    "prepare_model_for_qat",
    "calculate_pauc",
    "calculate_eer",
    "calculate_fah",
]
