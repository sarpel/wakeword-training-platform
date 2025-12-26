from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.config.defaults import WakewordConfig
from src.training.hpo import run_hpo
from src.training.hpo_results import HPOResult


class DummyDataset(Dataset):
    def __init__(self, size=20):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randn(1, 64, 50), torch.tensor(0)


def test_hpo_variable_expansion_and_result_structure():
    """
    Test that HPO optimizes new variables and returns a standardized HPOResult.
    """
    config = WakewordConfig()
    config.training.epochs = 1
    config.training.batch_size = 4
    config.data.n_mels = 64
    config.model.architecture = "lstm"  # Test LSTM specific params

    train_ds = DummyDataset(size=20)
    val_ds = DummyDataset(size=10)
    train_loader = DataLoader(train_ds, batch_size=4)
    val_loader = DataLoader(val_ds, batch_size=4)

    # Run HPO with expanded param groups
    # Include "Model" to check for hidden_size, num_layers (int, categorical)
    # Include "Augmentation" to check for pitch_shift_range (int)
    # Include "Loss" to check for focal_alpha (float)
    result = run_hpo(
        config,
        train_loader,
        val_loader,
        n_trials=2,  # Short run
        study_name="test_expanded_hpo",
        param_groups=["Model", "Augmentation", "Loss"],
        n_jobs=1,
        cache_dir=Path("cache/test_expanded_hpo"),
    )

    # 1. Check Result Structure
    assert isinstance(result, HPOResult)
    assert result.study_name == "test_expanded_hpo"
    assert isinstance(result.best_params, dict)

    # 2. Check Expanded Variables
    params = result.best_params

    # Model (LSTM)
    assert "hidden_size" in params
    assert "num_layers" in params
    assert isinstance(params["num_layers"], int)

    # Augmentation
    assert "pitch_shift_range" in params
    assert isinstance(params["pitch_shift_range"], int)

    # Loss
    assert "loss_function" in params
    if params["loss_function"] == "focal_loss":
        assert "focal_gamma" in params
        assert "focal_alpha" in params


if __name__ == "__main__":
    test_hpo_variable_expansion_and_result_structure()
