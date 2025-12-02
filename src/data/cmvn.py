"""
Corpus-level Cepstral Mean Variance Normalization (CMVN)
Implements global normalization statistics with persistence
"""

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

import structlog
import torch
import torch.nn as nn

from src.security import validate_path

logger = structlog.get_logger(__name__)


class CMVN(nn.Module):
    """
    Corpus-level Cepstral Mean Variance Normalization
    Inherits from nn.Module for seamless integration with models.
    """

    def __init__(self, stats_path: Optional[Path] = None, eps: float = 1e-8) -> None:
        """
        Initialize CMVN

        Args:
            stats_path: Path to stats.json file (for loading/saving)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.stats_path = Path(stats_path) if stats_path else None
        self.eps = eps

        # Register statistics as buffers (persistent state, not parameters)
        # We register them as None initially, but buffers must be tensors.
        # So we use register_buffer with a dummy tensor or handle loading logic carefully.
        # Better approach: register_buffer with empty tensor or None if allowed (it's not).
        # Strategy: use self.register_buffer("mean", torch.tensor(...)) only when we have values.
        # But __init__ must set up state.
        # Let's init as buffers if we can load them, otherwise placeholders.

        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("std", torch.zeros(1))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

        # Flag to check if initialized
        self._initialized = False

        # Load stats if path provided and exists
        if self.stats_path and self.stats_path.exists():
            self.load_stats()
            logger.info(f"Loaded CMVN stats from {self.stats_path}")

    def compute_stats(self, features_list: List[torch.Tensor], save: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute global mean and std from feature list

        Args:
            features_list: List of feature tensors (each can be (C, T) or (T, C))
            save: Whether to save stats to disk

        Returns:
            Tuple of (mean, std) tensors
        """
        logger.info(f"Computing CMVN stats from {len(features_list)} utterances")

        # Accumulate statistics
        sum_features = None
        sum_squared = None
        total_frames = 0

        for features in features_list:
            # Ensure features are 2D (C, T)
            if features.ndim == 3:
                # Batch dimension, squeeze it
                features = features.squeeze(0)

            # Convert to (C, T) if needed
            if features.shape[0] > features.shape[1]:
                features = features.transpose(0, 1)

            # Accumulate
            num_frames = features.shape[1]

            if sum_features is None:
                sum_features = features.sum(dim=1)
                sum_squared = (features**2).sum(dim=1)
            else:
                sum_features += features.sum(dim=1)
                sum_squared += (features**2).sum(dim=1)

            total_frames += num_frames

        # Compute global statistics
        # Ensure calculations are done on CPU initially to avoid VRAM usage for large accumulations
        # Or respect input device.
        assert sum_features is not None and sum_squared is not None
        mean = sum_features / total_frames
        variance = (sum_squared / total_frames) - (mean**2)
        std = torch.sqrt(variance.clamp(min=self.eps))

        # Update buffers
        self.mean = mean
        self.std = std
        self.count.fill_(total_frames)
        self._initialized = True

        logger.info(f"CMVN stats computed:")
        logger.info(f"  Total frames: {total_frames}")
        logger.info(f"  Feature dim: {self.mean.shape[0]}")
        logger.info(f"  Mean range: [{self.mean.min():.4f}, {self.mean.max():.4f}]")
        logger.info(f"  Std range: [{self.std.min():.4f}, {self.std.max():.4f}]")

        # Save if requested
        if save and self.stats_path:
            self.save_stats()

        return self.mean, self.std

    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply CMVN normalization to features (batch-aware)
        Args:
            features: Input features (B, C, T) or (C, T)
        """
        if not self._initialized:
            # If loaded via state_dict, check buffers
            if self.mean.numel() > 1:
                self._initialized = True
            else:
                raise RuntimeError("CMVN stats not computed. Call compute_stats() first.")

        # Handle batch dimension
        original_ndim = features.ndim
        if original_ndim == 2:  # (C, T)
            features = features.unsqueeze(0)

        # Normalize: (x - mean) / std
        # mean, std: (C,)
        # features: (B, C, T)
        mean = self.mean.view(1, -1, 1)
        std = self.std.view(1, -1, 1)

        normalized = (features - mean) / std

        if original_ndim == 2:
            normalized = normalized.squeeze(0)

        return normalized

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass alias for normalize"""
        return self.normalize(features)

    def denormalize(self, features: torch.Tensor) -> torch.Tensor:
        """Reverse CMVN normalization"""
        if not self._initialized:
            raise RuntimeError("CMVN stats not computed.")

        original_ndim = features.ndim
        if original_ndim == 2:
            features = features.unsqueeze(0)

        mean = self.mean.view(1, -1, 1)
        std = self.std.view(1, -1, 1)

        denormalized = (features * std) + mean

        if original_ndim == 2:
            denormalized = denormalized.squeeze(0)

        return denormalized

    def save_stats(self, path: Optional[Path] = None) -> None:
        """Save CMVN statistics to JSON"""
        save_path = Path(path) if path else self.stats_path
        if save_path is None:
            raise ValueError("No stats path provided")

        # Validate path
        save_path = validate_path(save_path)

        if not self._initialized:
            raise RuntimeError("No stats to save.")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        stats_dict = {
            "mean": self.mean.detach().cpu().numpy().tolist(),
            "std": self.std.detach().cpu().numpy().tolist(),
            "count": self.count.item(),
            "eps": self.eps,
        }

        with open(save_path, "w") as f:
            json.dump(stats_dict, f, indent=2)

        logger.info(f"CMVN stats saved to {save_path}")

    def load_stats(self, path: Optional[Path] = None) -> None:
        """Load CMVN statistics from JSON"""
        load_path = Path(path) if path else self.stats_path
        if load_path is None:
            raise ValueError("No stats path provided")

        # Validate path
        load_path = validate_path(load_path, must_exist=True, must_be_file=True)

        with open(load_path, "r") as f:
            stats_dict = json.load(f)

        # Load into buffers and move to device if module already on device (handled by register_buffer persistence logic usually, but here we set manually)
        # We must respect current device if possible, or let .to(device) handle it later.
        device = self.mean.device

        # Create new tensors from loaded data
        new_mean = torch.tensor(stats_dict["mean"], dtype=torch.float32, device=device)
        new_std = torch.tensor(stats_dict["std"], dtype=torch.float32, device=device)
        new_count = torch.tensor(stats_dict["count"], dtype=torch.long, device=device)

        # Update buffers
        # If shapes match, copy in-place to preserve memory/device
        if self.mean.shape == new_mean.shape:
            self.mean.copy_(new_mean)
            self.std.copy_(new_std)
            self.count.copy_(new_count)
        else:
            # If shapes differ (e.g. init with size 1, load size 64), re-register
            self.register_buffer("mean", new_mean)
            self.register_buffer("std", new_std)
            self.register_buffer("count", new_count)

        self.eps = stats_dict.get("eps", 1e-8)
        self._initialized = True

        logger.info(f"CMVN stats loaded from {load_path} (dim={self.mean.shape[0]})")


from torch.utils.data import Dataset


def compute_cmvn_from_dataset(dataset: Dataset, stats_path: Path, max_samples: Optional[int] = None) -> CMVN:
    """
    Compute CMVN statistics from a PyTorch dataset

    Args:
        dataset: PyTorch dataset that returns (features, label, metadata)
        stats_path: Path to save stats.json
        max_samples: Maximum number of samples to use (None = all)

    Returns:
        CMVN object with computed statistics
    """
    logger.info(f"Computing CMVN stats from dataset (size={len(dataset)})")  # type: ignore[arg-type]

    # Collect features
    features_list = []
    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)  # type: ignore[arg-type]

    for i in range(num_samples):
        features, _, _ = dataset[i]
        features_list.append(features)

        if (i + 1) % 1000 == 0:
            logger.info(f"Collected {i+1}/{num_samples} samples")

    # Create CMVN and compute stats
    cmvn = CMVN(stats_path=stats_path)
    cmvn.compute_stats(features_list, save=True)

    return cmvn


if __name__ == "__main__":
    # Test CMVN
    print("CMVN Test")
    print("=" * 60)

    # Create dummy features
    num_utterances = 100
    feature_dim = 64
    time_steps = 150

    features_list = []
    for i in range(num_utterances):
        # Random features with different means
        features = torch.randn(feature_dim, time_steps) + (i * 0.1)
        features_list.append(features)

    print(f"Created {num_utterances} dummy feature tensors ({feature_dim}x{time_steps})")

    # Compute CMVN stats
    cmvn = CMVN(stats_path=Path("test_cmvn_stats.json"))
    mean, std = cmvn.compute_stats(features_list)

    print(f"\nComputed stats:")
    print(f"  Mean: {mean[:5]} ...")
    print(f"  Std: {std[:5]} ...")

    # Test normalization
    test_features = torch.randn(feature_dim, time_steps)
    normalized = cmvn.normalize(test_features)

    print(f"\nTest normalization:")
    print(f"  Input mean: {test_features.mean():.4f}")
    print(f"  Normalized mean: {normalized.mean():.4f}")
    print(f"  Normalized std: {normalized.std():.4f}")

    # Test save/load
    cmvn.save_stats()
    print(f"\nStats saved to test_cmvn_stats.json")

    # Load in new object
    cmvn2 = CMVN(stats_path=Path("test_cmvn_stats.json"))

    # Verify loaded stats match
    assert torch.allclose(cmvn.mean, cmvn2.mean)
    assert torch.allclose(cmvn.std, cmvn2.std)
    print(f"✅ Save/load verification passed")

    print("\n✅ CMVN module test complete")
