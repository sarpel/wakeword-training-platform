"""
Corpus-level Cepstral Mean Variance Normalization (CMVN)
Implements global normalization statistics with persistence
"""
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class CMVN:
    """
    Corpus-level Cepstral Mean Variance Normalization

    Computes global mean and std across entire corpus and applies
    consistent normalization to train/val/test splits.
    """

    def __init__(
        self,
        stats_path: Optional[Path] = None,
        eps: float = 1e-8
    ):
        """
        Initialize CMVN

        Args:
            stats_path: Path to stats.json file (for loading/saving)
            eps: Small constant for numerical stability
        """
        self.stats_path = Path(stats_path) if stats_path else None
        self.eps = eps

        # Statistics
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.count: int = 0

        # Load stats if path provided and exists
        if self.stats_path and self.stats_path.exists():
            self.load_stats()
            logger.info(f"Loaded CMVN stats from {self.stats_path}")

    def compute_stats(
        self,
        features_list: list,
        save: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
                sum_squared = (features ** 2).sum(dim=1)
            else:
                sum_features += features.sum(dim=1)
                sum_squared += (features ** 2).sum(dim=1)

            total_frames += num_frames

        # Compute global statistics
        self.mean = sum_features / total_frames
        variance = (sum_squared / total_frames) - (self.mean ** 2)
        self.std = torch.sqrt(variance.clamp(min=self.eps))
        self.count = total_frames

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
        Apply CMVN normalization to features

        Args:
            features: Input features (B, C, T) or (C, T)

        Returns:
            Normalized features (same shape as input)
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("CMVN stats not computed. Call compute_stats() first.")

        # Move stats to same device as features
        mean = self.mean.to(features.device)
        std = self.std.to(features.device)

        # Handle batch dimension
        original_shape = features.shape
        if features.ndim == 2:
            # (C, T) -> add batch dim
            features = features.unsqueeze(0)

        # Normalize: (x - mean) / std
        # mean and std are (C,), features are (B, C, T)
        mean = mean.view(1, -1, 1)
        std = std.view(1, -1, 1)

        normalized = (features - mean) / std

        # Restore original shape
        if len(original_shape) == 2:
            normalized = normalized.squeeze(0)

        return normalized

    def denormalize(self, features: torch.Tensor) -> torch.Tensor:
        """
        Reverse CMVN normalization

        Args:
            features: Normalized features

        Returns:
            Original scale features
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("CMVN stats not computed.")

        mean = self.mean.to(features.device)
        std = self.std.to(features.device)

        original_shape = features.shape
        if features.ndim == 2:
            features = features.unsqueeze(0)

        mean = mean.view(1, -1, 1)
        std = std.view(1, -1, 1)

        denormalized = (features * std) + mean

        if len(original_shape) == 2:
            denormalized = denormalized.squeeze(0)

        return denormalized

    def save_stats(self, path: Optional[Path] = None):
        """
        Save CMVN statistics to JSON

        Args:
            path: Path to save stats (uses self.stats_path if None)
        """
        save_path = Path(path) if path else self.stats_path

        if save_path is None:
            raise ValueError("No stats path provided")

        if self.mean is None or self.std is None:
            raise RuntimeError("No stats to save. Compute stats first.")

        # Create parent directory
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to lists for JSON serialization
        stats_dict = {
            'mean': self.mean.cpu().numpy().tolist(),
            'std': self.std.cpu().numpy().tolist(),
            'count': self.count,
            'eps': self.eps
        }

        with open(save_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)

        logger.info(f"CMVN stats saved to {save_path}")

    def load_stats(self, path: Optional[Path] = None):
        """
        Load CMVN statistics from JSON

        Args:
            path: Path to load stats from (uses self.stats_path if None)
        """
        load_path = Path(path) if path else self.stats_path

        if load_path is None:
            raise ValueError("No stats path provided")

        if not load_path.exists():
            raise FileNotFoundError(f"Stats file not found: {load_path}")

        with open(load_path, 'r') as f:
            stats_dict = json.load(f)

        self.mean = torch.tensor(stats_dict['mean'], dtype=torch.float32)
        self.std = torch.tensor(stats_dict['std'], dtype=torch.float32)
        self.count = stats_dict['count']
        self.eps = stats_dict.get('eps', 1e-8)

        logger.info(f"CMVN stats loaded from {load_path}")
        logger.info(f"  Feature dim: {self.mean.shape[0]}")
        logger.info(f"  Total frames: {self.count}")


def compute_cmvn_from_dataset(
    dataset,
    stats_path: Path,
    max_samples: Optional[int] = None
) -> CMVN:
    """
    Compute CMVN statistics from a PyTorch dataset

    Args:
        dataset: PyTorch dataset that returns (features, label, metadata)
        stats_path: Path to save stats.json
        max_samples: Maximum number of samples to use (None = all)

    Returns:
        CMVN object with computed statistics
    """
    logger.info(f"Computing CMVN stats from dataset (size={len(dataset)})")

    # Collect features
    features_list = []
    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)

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
