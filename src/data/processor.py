"""
GPU-based Audio Processor for Training Pipeline
Handles audio processing, feature extraction, and CMVN normalization on GPU.

This module is designed for efficient batch processing during training,
combining waveform augmentation and feature extraction in a single pass.
"""
from pathlib import Path
from typing import Optional

import structlog
import torch
import torch.nn as nn

from src.data.cmvn import CMVN
from src.data.feature_extraction import FeatureExtractor

logger = structlog.get_logger(__name__)


class AudioProcessor(nn.Module):
    """
    GPU-based audio processor for training pipeline.
    
    Combines feature extraction and optional CMVN normalization
    in a single nn.Module for efficient GPU processing.
    """

    def __init__(
        self,
        config,
        cmvn_path: Optional[Path] = None,
        device: str = "cuda",
    ):
        """
        Initialize GPU audio processor.

        Args:
            config: WakewordConfig with data parameters
            cmvn_path: Path to CMVN stats.json (or None to disable)
            device: Device for processing ('cuda' or 'cpu')
        """
        super().__init__()
        
        self.sample_rate = config.data.sample_rate
        self.audio_duration = config.data.audio_duration
        self.target_samples = int(self.sample_rate * self.audio_duration)
        self.device_name = device  # Store for reference
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            sample_rate=config.data.sample_rate,
            feature_type=config.data.feature_type,
            n_mels=config.data.n_mels,
            n_mfcc=config.data.n_mfcc,
            n_fft=config.data.n_fft,
            hop_length=config.data.hop_length,
            device=device,
        )
        
        # CMVN normalization (optional)
        self.cmvn = None
        if cmvn_path is not None and Path(cmvn_path).exists():
            self.cmvn = CMVN(stats_path=cmvn_path)
            logger.info(f"CMVN loaded from {cmvn_path}")
        else:
            logger.info("CMVN disabled (no stats path provided or file not found)")
        
        # Move to device
        if device and device != "cpu":
            self.to(device)
            
        logger.info(f"AudioProcessor initialized on {device}")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Process audio waveform to features.
        
        Handles input normalization, feature extraction, and CMVN.

        Args:
            waveform: Input waveform tensor
                - Shape: (B, S) for batch of 1D waveforms
                - Shape: (B, 1, S) for batch of mono waveforms
                - Shape: (S,) for single waveform

        Returns:
            Features tensor with shape (B, 1, n_mels, time_steps) or (B, 1, n_mfcc, time_steps)
        """
        # Standardize input shape
        original_ndim = waveform.ndim
        
        if waveform.ndim == 1:
            # Single waveform (S,) -> (1, S)
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 3:
            # (B, 1, S) -> (B, S)
            waveform = waveform.squeeze(1)
        # Now waveform is (B, S)
        
        # Normalize length to target samples
        waveform = self._normalize_length(waveform)
        
        # Extract features
        # feature_extractor expects (B, S) and returns (B, 1, freq, time)
        features = self.feature_extractor(waveform)
        
        # Apply CMVN if available
        if self.cmvn is not None:
            # CMVN expects (B, C, T) but features are (B, 1, H, W)
            # We need to handle this - squeeze channel, apply CMVN, unsqueeze
            # Actually, looking at CMVN.normalize(), it handles (B, C, T)
            # Our features are (B, 1, n_mels, time)
            # Squeeze channel: (B, n_mels, time)
            batch_size = features.shape[0]
            features_squeezed = features.squeeze(1)  # (B, n_mels, time)
            features_normalized = self.cmvn.normalize(features_squeezed)
            features = features_normalized.unsqueeze(1)  # (B, 1, n_mels, time)
        
        # If input was single waveform, remove batch dimension
        if original_ndim == 1:
            features = features.squeeze(0)
            
        return features

    def _normalize_length(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Normalize waveform length to target samples.

        Args:
            waveform: Input waveform (B, S)

        Returns:
            Length-normalized waveform (B, target_samples)
        """
        current_samples = waveform.shape[-1]
        
        if current_samples == self.target_samples:
            return waveform
        elif current_samples > self.target_samples:
            # Trim from center
            start = (current_samples - self.target_samples) // 2
            return waveform[..., start : start + self.target_samples]
        else:
            # Pad with zeros
            pad_amount = self.target_samples - current_samples
            pad_left = pad_amount // 2
            pad_right = pad_amount - pad_left
            # F.pad expects (left, right) for last dimension
            return torch.nn.functional.pad(waveform, (pad_left, pad_right), mode='constant', value=0)

    def get_output_shape(self, input_samples: int = None) -> tuple:
        """
        Get output feature shape.

        Args:
            input_samples: Number of input samples (default: target_samples)

        Returns:
            Output feature shape (1, n_mels, time_steps)
        """
        if input_samples is None:
            input_samples = self.target_samples
        return self.feature_extractor.get_output_shape(input_samples)


if __name__ == "__main__":
    # Test AudioProcessor
    print("AudioProcessor Test")
    print("=" * 60)

    # Create mock config
    class MockDataConfig:
        sample_rate = 16000
        audio_duration = 1.5
        feature_type = "mel"
        n_mels = 64
        n_mfcc = 0
        n_fft = 400
        hop_length = 160

    class MockConfig:
        data = MockDataConfig()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize processor
    processor = AudioProcessor(
        config=MockConfig(),
        cmvn_path=None,
        device=device,
    )

    # Test with single waveform
    single_waveform = torch.randn(24000).to(device)
    print(f"\nSingle waveform input shape: {single_waveform.shape}")
    
    single_features = processor(single_waveform)
    print(f"Single output shape: {single_features.shape}")

    # Test with batch
    batch_waveform = torch.randn(8, 24000).to(device)
    print(f"\nBatch waveform input shape: {batch_waveform.shape}")
    
    batch_features = processor(batch_waveform)
    print(f"Batch output shape: {batch_features.shape}")

    # Test with (B, 1, S) input
    batch_3d = torch.randn(4, 1, 24000).to(device)
    print(f"\n3D waveform input shape: {batch_3d.shape}")
    
    batch_3d_features = processor(batch_3d)
    print(f"3D output shape: {batch_3d_features.shape}")

    print("\n✅ AudioProcessor test complete")
