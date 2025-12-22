
import time

from pathlib import Path
from src.config.defaults import WakewordConfig
from src.training.hpo import run_hpo
from src.data.dataset import WakewordDataset
from torch.utils.data import DataLoader

def benchmark_hpo_sweep():
    """
    Benchmark the current HPO implementation.
    """
    print("\nStarting HPO Benchmark...")
    
    # 1. Setup Config
    config = WakewordConfig()
    config.training.epochs = 2  # Keep it short for benchmark
    config.training.batch_size = 32
    config.data.use_precomputed_features_for_training = False # Force feature extraction logic if needed
    
    # 2. Mock Data Loaders (Empty or minimal for speed)
    # We need to mock the dataset to avoid loading actual audio files which takes time and requires data
    # But for a realistic benchmark of the *loop*, we need the structure.
    # However, since we might not have the full dataset in this env, let's create a Dummy Dataset.
    
    from torch.utils.data import Dataset
    import torch
    
    class DummyDataset(Dataset):
        def __init__(self, size=100):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            # Return random features and label
            # Mel spec shape: (1, n_mels, time_steps)
            # Default n_mels=80, duration=1s -> approx 101 frames
            return torch.randn(1, 80, 101), torch.tensor(0)

    train_ds = DummyDataset(size=200)
    val_ds = DummyDataset(size=50)
    
    train_loader = DataLoader(train_ds, batch_size=32)
    val_loader = DataLoader(val_ds, batch_size=32)
    
    # 3. Run HPO
    start_time = time.time()
    study = run_hpo(
        config,
        train_loader,
        val_loader,
        n_trials=5, # Run 5 trials
        study_name="benchmark_hpo",
        n_jobs=1,
        cache_dir=Path("cache/benchmark_hpo")
    )
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"\nBenchmark Complete.")
    print(f"Total Time: {duration:.2f} seconds")
    print(f"Time per trial: {duration/5:.2f} seconds")
    
    return duration

if __name__ == "__main__":
    benchmark_hpo_sweep()
