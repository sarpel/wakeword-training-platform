import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.config.defaults import WakewordConfig
from src.training.hpo import run_hpo


class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randn(1, 80, 101), torch.tensor(0)


def benchmark_hpo_sweep():
    """
    Benchmark the current HPO implementation.
    """
    print("\nStarting HPO Benchmark...")

    # 1. Setup Config
    config = WakewordConfig()
    config.training.epochs = 2
    config.training.batch_size = 32
    config.training.num_workers = 0
    config.data.use_precomputed_features_for_training = False

    train_ds = DummyDataset(size=200)
    val_ds = DummyDataset(size=50)
    train_loader = DataLoader(train_ds, batch_size=32)
    val_loader = DataLoader(val_ds, batch_size=32)

    # 3. Run HPO
    start_time = time.time()
    result = run_hpo(
        config,
        train_loader,
        val_loader,
        n_trials=6,
        study_name="benchmark_hpo",
        n_jobs=2,
        cache_dir=Path("cache/benchmark_hpo"),
    )
    end_time = time.time()

    duration = end_time - start_time
    print(f"\nBenchmark Complete.")
    print(f"Total Time: {duration:.2f} seconds")
    print(f"Time per trial: {duration/6:.2f} seconds")

    return duration


if __name__ == "__main__":
    benchmark_hpo_sweep()
