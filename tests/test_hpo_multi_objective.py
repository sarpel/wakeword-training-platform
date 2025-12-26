from pathlib import Path
from unittest.mock import MagicMock

import optuna
import pytest
import torch
import torch.nn as nn

from src.config.defaults import WakewordConfig
from src.training.hpo import Objective


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


def test_objective_multi_objective():
    config = WakewordConfig()
    config.training.epochs = 1
    config.training.batch_size = 2

    # Mock loaders
    train_loader = MagicMock()
    train_loader.dataset = [(torch.randn(1, 10), 0)] * 4
    val_loader = MagicMock()
    val_loader.dataset = [(torch.randn(1, 10), 0)] * 4

    # Mock Trainer
    with MagicMock() as mock_trainer_cls:
        # We need to mock Trainer within Objective
        import src.training.hpo

        src.training.hpo.Trainer = MagicMock()
        mock_trainer = src.training.hpo.Trainer.return_value

        # Mock validation results
        mock_trainer.train.return_value = {"best_val_f1": 0.8, "best_val_fpr": 0.01}

        # Mock metrics tracker
        mock_metrics = MagicMock()
        mock_metrics.pauc = 0.85
        mock_metrics.latency_ms = 5.0
        mock_trainer.val_metrics_tracker.get_best_epoch.return_value = (0, mock_metrics)

        objective = Objective(config, train_loader, val_loader, n_jobs=1)

        study = optuna.create_study(directions=["maximize", "minimize"])
        trial = study.ask()

        results = objective(trial)

        assert isinstance(results, tuple)
        assert len(results) == 2
        assert results[0] == 0.85  # pAUC
        assert results[1] == 5.0  # Latency
