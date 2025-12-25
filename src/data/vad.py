"""
Voice Activity Detection (VAD) Utilities
Implements energy-based VAD for dataset cleaning and preprocessing.
"""

from typing import Any, List, Optional

import structlog
import torch

logger = structlog.get_logger(__name__)


class EnergyVAD:
    """
    Simple Energy-based Voice Activity Detector.
    Used to filter out silent segments or noise-only files.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length_ms: int = 30,
        frame_shift_ms: int = 10,
        energy_threshold: float = 0.05,
    ):
        """
        Args:
            sample_rate: Audio sample rate
            frame_length_ms: Length of analysis frame in ms
            frame_shift_ms: Shift between frames in ms
            energy_threshold: Threshold for detecting voice (0.0 to 1.0)
                              Relative to signal max or absolute?
                              We'll use relative to signal peak for robustness.
        """
        self.sample_rate = sample_rate
        self.frame_length = int(sample_rate * frame_length_ms / 1000)
        self.frame_shift = int(sample_rate * frame_shift_ms / 1000)
        self.energy_threshold = energy_threshold

    def is_speech(self, waveform: torch.Tensor) -> bool:
        """
        Check if waveform contains speech.

        Args:
            waveform: (1, T) or (T,) tensor

        Returns:
            True if speech detected, False otherwise
        """
        if waveform.ndim == 2:
            waveform = waveform.squeeze(0)

        if waveform.numel() < self.frame_length:
            return False

        # Normalize
        max_val = waveform.abs().max()
        if max_val < 1e-6:
            return False

        normalized = waveform / max_val

        # Frame blocking (sliding window)
        # shape: (num_frames, frame_length)
        frames = normalized.unfold(0, self.frame_length, self.frame_shift)

        # Compute energy
        # Mean square of each frame
        energies = frames.pow(2).mean(dim=1)

        # Check if any frame exceeds threshold
        # We require a minimum duration of speech (e.g., 3 consecutive frames)
        mask = energies > self.energy_threshold

        # Check for consecutive frames
        # We can use convolution for this
        consecutive_frames = 3
        if len(mask) < consecutive_frames:
            return bool(mask.any().item())

        kernel = torch.ones(consecutive_frames, device=mask.device)
        # conv1d needs (N, C, L)
        mask_float = mask.float().view(1, 1, -1)
        kernel = kernel.view(1, 1, -1)

        convolved = torch.nn.functional.conv1d(mask_float, kernel)

        # If any value == consecutive_frames, we have a sequence
        return bool((convolved == consecutive_frames).any().item())

    def filter_dataset(self, dataset: List[Any], threshold: Optional[float] = None) -> List[Any]:
        """
        Filter a dataset (list of files) using VAD.
        Not implemented here as it depends on dataset structure.
        """
        return dataset  # Placeholder implementation
