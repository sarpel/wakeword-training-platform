import random
import sys
import unittest
from pathlib import Path

import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.augmentation import AudioAugmentation


class TestExhaustiveCoverage(unittest.TestCase):
    def test_noise_coverage(self):
        # Mock files
        mock_files = [Path(f"noise_{i}.wav") for i in range(10)]

        # Initialize with small buffer size
        aug = AudioAugmentation(background_noise_files=mock_files, max_background_noises=3)

        seen_files = set()

        # Cycle 1: Should get 3 unique files
        # (Already loaded in __init__)
        # We can't easily check 'selected_files' directly as it's local
        # But we can check internal _remaining list size
        self.assertEqual(len(aug._remaining_background_noise_files), 7)

        # Cycle 2: Get next 3
        aug._load_background_noises_subset()
        self.assertEqual(len(aug._remaining_background_noise_files), 4)

        # Cycle 3: Get next 3
        aug._load_background_noises_subset()
        self.assertEqual(len(aug._remaining_background_noise_files), 1)

        # Cycle 4: Get next 3 (requires refill)
        # Pool has 1. We need 3.
        # It takes 1. Pool becomes empty.
        # Refill happens. Shuffles 10.
        # Takes 2 more. Pool becomes 8.
        aug._load_background_noises_subset()
        self.assertEqual(len(aug._remaining_background_noise_files), 8)

        print("Exhaustive coverage sequence verified successfully!")


if __name__ == "__main__":
    # Simplified manual test since we don't want to actually load wav files
    mock_files = [Path(f"noise_{i}.wav") for i in range(10)]

    # We patch torchaudio.load to avoid file errors
    import torchaudio

    orig_load = torchaudio.load
    torchaudio.load = lambda x: (torch.randn(1, 16000), 16000)

    try:
        tester = TestExhaustiveCoverage()
        tester.test_noise_coverage()
        print("✅ Logic test passed!")
    finally:
        torchaudio.load = orig_load
