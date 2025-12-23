"""
Dataset Configuration Validator
Checks for consistency between configuration and dataset features.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import structlog

from src.config.defaults import WakewordConfig
from src.data.feature_extraction import FeatureExtractor

logger = structlog.get_logger(__name__)


class DatasetConfigValidator:
    """Validates configuration against dataset features"""

    def __init__(self) -> None:
        """Initialize validator"""
        pass

    def calculate_expected_shape(self, config: WakewordConfig) -> Tuple[int, int, int]:
        """
        Calculate expected feature shape based on configuration

        Args:
            config: Configuration

        Returns:
            Tuple of (channels, freq, time)
        """
        # Create a temporary feature extractor to get the shape
        # We only need the shape, so we don't need a real device
        extractor = FeatureExtractor(
            sample_rate=config.data.sample_rate,
            feature_type=config.data.feature_type,
            n_mels=config.data.n_mels,
            n_mfcc=config.data.n_mfcc,
            n_fft=config.data.n_fft,
            hop_length=config.data.hop_length,
            device="cpu",
        )

        num_samples = int(config.data.sample_rate * config.data.audio_duration)
        return extractor.get_output_shape(num_samples)

    def validate_dataset_features(self, config: WakewordConfig, npy_dir: Path) -> List[Dict[str, Any]]:
        """
        Validate configuration against existing .npy features

        Args:
            config: Configuration to validate
            npy_dir: Directory containing .npy files

        Returns:
            List of mismatch details
        """
        npy_dir = Path(npy_dir)
        if not npy_dir.exists():
            return []

        expected_shape = self.calculate_expected_shape(config)
        mismatches = []

        # Scan for .npy files
        npy_files = list(npy_dir.rglob("*.npy"))

        # To avoid performance issues with large datasets, we sample or check first few
        # But for this utility, checking a few from each category might be enough
        # For now, let's check all but limited to a reasonable number if needed

        for npy_file in npy_files[:100]:  # Check first 100 files
            try:
                data = np.load(str(npy_file), mmap_mode="r")
                actual_shape = data.shape

                # Handle 2D vs 3D (missing channel dim)
                if len(actual_shape) == 2 and len(expected_shape) == 3:
                    if actual_shape == expected_shape[1:]:
                        continue  # Match

                if actual_shape != expected_shape:
                    mismatches.append(
                        {
                            "file": str(npy_file),
                            "issue": "shape_mismatch",
                            "actual_shape": actual_shape,
                            "expected_shape": expected_shape,
                        }
                    )
                    # If we found a mismatch, we can stop early if we just want to know IF there is one
                    # but let's return some details.
                    if len(mismatches) >= 5:
                        break
            except Exception as e:
                logger.error(f"Error validating {npy_file}: {e}")

        return mismatches
