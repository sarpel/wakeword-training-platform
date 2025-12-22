
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from src.config.defaults import WakewordConfig
from src.training.hpo import run_hpo
from pathlib import Path

class DummyDataset(Dataset):
    def __init__(self, size=50):
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return torch.randn(1, 64, 100), torch.tensor(0)

def test_hpo_parallel_execution():
    """
    Test that HPO can run with n_jobs > 1 without crashing.
    """
    config = WakewordConfig()
    config.training.epochs = 1
    config.training.batch_size = 16
    config.data.n_mels = 64
    
    # Setup dummy data
    train_ds = DummyDataset(size=50)
    val_ds = DummyDataset(size=20)
    train_loader = DataLoader(train_ds, batch_size=16)
    val_loader = DataLoader(val_ds, batch_size=16)
    
    # Run with n_jobs=2
    # This should fail or be unsafe in current implementation due to DynamicBatchSampler state sharing
    try:
        study = run_hpo(
            config,
            train_loader,
            val_loader,
            n_trials=4,
            study_name="test_parallel_hpo",
            n_jobs=2,
            cache_dir=Path("cache/test_parallel_hpo")
        )
        assert len(study.trials) == 4
        assert study.best_value is not None
    except Exception as e:
        pytest.fail(f"Parallel HPO failed: {e}")

if __name__ == "__main__":
    test_hpo_parallel_execution()
