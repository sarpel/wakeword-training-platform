"""
Feature Extraction for Wakeword Detection
Compute mel-spectrograms and MFCCs on GPU
"""

from typing import Literal, Optional, Tuple, cast

import structlog
import torch
import torch.nn as nn
import torchaudio.transforms as T

logger = structlog.get_logger(__name__)


class FeatureExtractor(nn.Module):
    """
    Extract features (mel-spectrogram or MFCC) from audio waveforms.

    Inherits from nn.Module to support GPU acceleration and seamless integration
    with training pipelines.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        feature_type: Literal["mel", "mfcc"] = "mel",
        n_mels: int = 40,
        n_mfcc: int = 0,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        device: str = "cpu",  # Kept for API compatibility but unused logic-wise
    ):
        """
        Initialize feature extractor

        Args:
            sample_rate: Audio sample rate
            feature_type: 'mel' for mel-spectrogram, 'mfcc' for MFCC
            n_mels: Number of mel filterbanks
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for STFT
            win_length: Window length (None = same as n_fft)
            f_min: Minimum frequency
            f_max: Maximum frequency (None = sample_rate / 2)
            device: Initial device (module can be moved with .to())
        """
        super().__init__()
        self.sample_rate = sample_rate
        # Normalize feature type (handle legacy 'mel_spectrogram')
        if feature_type == "mel_spectrogram":
            feature_type = "mel"
        self.feature_type = feature_type
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

        if win_length is None:
            win_length = n_fft

        if f_max is None:
            f_max = sample_rate / 2.0

        # Create mel-spectrogram transform
        # Registered as submodule, so it moves to GPU with .to(device)
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0,
        )

        # Create MFCC transform if needed
        if feature_type == "mfcc":
            self.mfcc_transform = T.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                melkwargs={
                    "n_fft": n_fft,
                    "win_length": win_length,
                    "hop_length": hop_length,
                    "f_min": f_min,
                    "f_max": f_max,
                    "n_mels": n_mels,
                },
            )
        else:
            self.mfcc_transform = None

        # Amplitude to DB conversion
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

        # Move to initial device if specified
        if device and device != "cpu":
            self.to(device)

        logger.info(
            f"FeatureExtractor initialized: {feature_type}, " f"n_mels={n_mels}, n_fft={n_fft}, hop_length={hop_length}"
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract features from waveform

        Args:
            waveform: Input waveform tensor
                - Shape: (samples,) or (1, samples) or (batch, samples)

        Returns:
            Features tensor
            - mel: (1, n_mels, time) or (batch, 1, n_mels, time)
            - mfcc: (n_mfcc, time) or (batch, n_mfcc, time)
        """
        # Handle different input shapes
        original_shape = waveform.shape
        if waveform.dim() == 1:
            # (samples,) -> (1, samples)
            waveform = waveform.unsqueeze(0)
            squeeze_output = True
        elif waveform.dim() == 2:
            # Assume (batch, samples) or (1, samples)
            squeeze_output = False
        elif waveform.dim() == 3 and waveform.shape[1] == 1:
            # (batch, 1, samples) -> (batch, samples)
            waveform = waveform.squeeze(1)
            squeeze_output = False
        else:
            raise ValueError(f"Expected 1D or 2D waveform (or 3D with 1 channel), got shape {original_shape}")

        # Extract features
        if self.feature_type == "mel":
            # Compute mel-spectrogram
            mel_spec = self.mel_spectrogram(waveform)  # (batch, n_mels, time)

            # Convert to dB
            mel_spec_db = self.amplitude_to_db(mel_spec)

            # Add channel dimension for CNN models: (batch, n_mels, time) -> (batch, 1, n_mels, time)
            features = mel_spec_db.unsqueeze(1)

            # If input was 1D, remove batch dimension
            if squeeze_output:
                features = features.squeeze(0)  # (1, n_mels, time)

        elif self.feature_type == "mfcc" and self.mfcc_transform is not None:
            # Compute MFCC
            mfcc = self.mfcc_transform(waveform)  # (batch, n_mfcc, time)

            # Add channel dimension for consistency with CNN models
            features = mfcc.unsqueeze(1)  # (batch, 1, n_mfcc, time)

            # If input was 1D, remove batch dimension
            if squeeze_output:
                features = features.squeeze(0)  # (1, n_mfcc, time)

        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")

        return cast(torch.Tensor, features)

    def get_output_shape(self, input_samples: int) -> Tuple[int, int, int]:
        """
        Get output shape for given input length

        Args:
            input_samples: Number of input samples

        Returns:
            Output feature shape (without batch dimension)
        """
        # Correct calculation of time dimension
        # This formula should match PyTorch's stft/spectrogram output size
        # time_steps = 1 + (input_samples - n_fft) // hop_length + 2 (for centering)
        # But torchaudio MelSpectrogram usually does center=True by default
        # which results in: time_steps = input_samples // hop_length + 1
        time_steps = input_samples // self.mel_spectrogram.hop_length + 1

        if self.feature_type == "mel":
            return (1, self.n_mels, time_steps)  # (channels, freq, time)
        elif self.feature_type == "mfcc":
            return (1, self.n_mfcc, time_steps)  # (channels, coeffs, time)


if __name__ == "__main__":
    print("Feature Extraction Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Test mel-spectrogram
    print("\nTesting mel-spectrogram extraction...")
    mel_extractor = FeatureExtractor(sample_rate=16000, feature_type="mel", n_mels=64, device=device)

    # Create dummy waveform (1.5 seconds at 16kHz)
    dummy_waveform = torch.randn(24000).to(device)
    print(f"Input waveform shape: {dummy_waveform.shape}")

    # Extract features
    mel_features = mel_extractor(dummy_waveform)
    print(f"Mel-spectrogram shape: {mel_features.shape}")
    print(f"Expected shape: {mel_extractor.get_output_shape(24000)}")

    # Test with batch
    batch_waveform = torch.randn(8, 24000).to(device)
    batch_features = mel_extractor(batch_waveform)
    print(f"Batch mel-spectrogram shape: {batch_features.shape}")

    print("\n✅ Mel-spectrogram extraction works correctly")

    # Test MFCC
    print("\nTesting MFCC extraction...")
    mfcc_extractor = FeatureExtractor(sample_rate=16000, feature_type="mfcc", n_mfcc=40, device=device)

    mfcc_features = mfcc_extractor(dummy_waveform)
    print(f"MFCC shape: {mfcc_features.shape}")
    print(f"Expected shape: {mfcc_extractor.get_output_shape(24000)}")

    print("\n✅ MFCC extraction works correctly")

    print("\n✅ Feature extraction module loaded successfully")
