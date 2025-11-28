"""
GPU-based Audio Processing Pipeline
Handles feature extraction and CMVN normalization on GPU for maximum performance
"""
from pathlib import Path
from typing import Optional

import structlog
import torch
import torch.nn as nn
import torchaudio.transforms as T

from src.data.cmvn import CMVN

logger = structlog.get_logger(__name__)


class AudioProcessor(nn.Module):
    """
    GPU-accelerated audio processing pipeline
    Converts raw audio waveforms to normalized features on GPU

    This module is designed to replace CPU-based feature extraction in the dataset
    by moving all processing to GPU for better performance.
    """

    def __init__(
        self,
        config,
        cmvn_path: Optional[Path] = None,
        device: str = "cuda",
    ):
        """
        Initialize GPU audio processor

        Args:
            config: WakewordConfig object with audio parameters
            cmvn_path: Path to CMVN stats file (optional)
            device: Device for processing (default: 'cuda')
        """
        super().__init__()

        self.config = config
        self.device = device
        self.sample_rate = config.data.sample_rate

        # Feature extraction parameters
        self.n_fft = config.data.n_fft
        self.hop_length = config.data.hop_length
        self.n_mels = config.data.n_mels
        self.feature_type = config.data.feature_type

        # Normalize feature type (handle legacy 'mel_spectrogram')
        if self.feature_type == "mel_spectrogram":
            self.feature_type = "mel"

        # Create mel spectrogram transform (GPU-compatible)
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
            normalized=False,
        ).to(device)

        # Optional MFCC transform
        self.mfcc_transform = None
        if self.feature_type == "mfcc" and config.data.n_mfcc > 0:
            self.mfcc_transform = T.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=config.data.n_mfcc,
                melkwargs={
                    'n_fft': self.n_fft,
                    'hop_length': self.hop_length,
                    'n_mels': self.n_mels,
                }
            ).to(device)

        # Load CMVN if provided
        self.cmvn = None
        if cmvn_path and Path(cmvn_path).exists():
            self.cmvn = CMVN(stats_path=cmvn_path).to(device)
            logger.info(f"AudioProcessor: Loaded CMVN from {cmvn_path}")

        logger.info(
            f"AudioProcessor initialized on {device}: "
            f"feature_type={self.feature_type}, "
            f"n_mels={self.n_mels}, "
            f"cmvn={'enabled' if self.cmvn else 'disabled'}"
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Process raw audio waveform to features

        Args:
            waveform: Input audio tensor
                     - Shape: (batch, samples) or (batch, 1, samples) or (batch, channels, samples)

        Returns:
            Feature tensor: (batch, 1, n_mels, time_steps)
        """
        # Ensure waveform is on correct device
        if waveform.device != self.device:
            waveform = waveform.to(self.device)

        # Handle different input shapes
        # Expected: (batch, samples) or (batch, 1, samples)
        if waveform.ndim == 1:
            # Single audio: (samples,) -> (1, samples)
            waveform = waveform.unsqueeze(0)

        if waveform.ndim == 2:
            # (batch, samples) -> already correct for MelSpectrogram
            pass
        elif waveform.ndim == 3:
            # (batch, 1, samples) or (batch, channels, samples)
            # Squeeze channel dimension if mono
            if waveform.shape[1] == 1:
                waveform = waveform.squeeze(1)  # (batch, samples)
            else:
                # Multi-channel: convert to mono by averaging
                waveform = waveform.mean(dim=1)  # (batch, samples)
        else:
            raise ValueError(
                f"Unexpected waveform shape: {waveform.shape}. "
                f"Expected (batch, samples) or (batch, 1, samples)"
            )

        # Extract features based on type
        if self.feature_type == "mfcc" and self.mfcc_transform is not None:
            # MFCC extraction
            features = self.mfcc_transform(waveform)  # (batch, n_mfcc, time)
        else:
            # Mel spectrogram extraction (default)
            features = self.mel_spectrogram(waveform)  # (batch, n_mels, time)

            # Convert to log scale (with epsilon for numerical stability)
            features = torch.log(features + 1e-9)

        # Add channel dimension: (batch, n_mels, time) -> (batch, 1, n_mels, time)
        features = features.unsqueeze(1)

        # Apply CMVN normalization if loaded
        if self.cmvn is not None:
            # CMVN expects (batch, channels, time), so we need to handle this
            # Current: (batch, 1, n_mels, time)
            # CMVN normalize expects (batch, channels, time) where channels = n_mels
            # So we need to reshape before and after

            batch_size, _, n_mels, time_steps = features.shape
            # Reshape to (batch, n_mels, time)
            features_for_cmvn = features.squeeze(1)  # (batch, n_mels, time)

            # Apply CMVN: (batch, n_mels, time) -> (batch, n_mels, time)
            features_normalized = self.cmvn(features_for_cmvn)

            # Reshape back: (batch, n_mels, time) -> (batch, 1, n_mels, time)
            features = features_normalized.unsqueeze(1)

        return features


if __name__ == "__main__":
    # Test AudioProcessor
    print("AudioProcessor Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create test config
    from src.config.defaults import WakewordConfig
    config = WakewordConfig()

    # Create processor
    processor = AudioProcessor(
        config=config,
        cmvn_path=None,  # No CMVN for basic test
        device=device
    )
    print(f"✅ AudioProcessor created")

    # Test with dummy audio
    # Simulate 1.5 seconds of audio at 16kHz = 24000 samples
    batch_size = 4
    samples = int(config.data.sample_rate * config.data.audio_duration)
    test_audio = torch.randn(batch_size, samples).to(device)

    print(f"\nTest input shape: {test_audio.shape}")

    # Process
    features = processor(test_audio)
    print(f"Output features shape: {features.shape}")
    print(f"Expected: (batch={batch_size}, channels=1, n_mels={config.data.n_mels}, time_steps)")

    # Verify shape
    assert features.shape[0] == batch_size
    assert features.shape[1] == 1  # Channel dimension
    assert features.shape[2] == config.data.n_mels
    print(f"✅ Shape verification passed")

    # Test with different input shapes
    print(f"\nTesting different input shapes:")

    # (batch, 1, samples)
    test_audio_3d = torch.randn(batch_size, 1, samples).to(device)
    features_3d = processor(test_audio_3d)
    print(f"  Input (B, 1, S): {test_audio_3d.shape} -> Output: {features_3d.shape}")
    assert features_3d.shape == features.shape

    # Single audio (samples,)
    test_audio_1d = torch.randn(samples).to(device)
    features_1d = processor(test_audio_1d)
    print(f"  Input (S,): {test_audio_1d.shape} -> Output: {features_1d.shape}")
    assert features_1d.shape[0] == 1  # Batch size 1

    print(f"\n✅ All AudioProcessor tests passed")
    print("AudioProcessor module ready for use")
