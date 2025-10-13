"""
Audio Data Augmentation Pipeline
GPU-accelerated augmentation using torchaudio and torch operations
"""
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import random
import logging

logger = logging.getLogger(__name__)


class AudioAugmentation:
    """
    CPU-based audio augmentation pipeline for dataset preprocessing
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        device: str = "cpu",  # Audio augmentation always on CPU
        # Time domain
        time_stretch_range: Tuple[float, float] = (0.9, 1.1),
        pitch_shift_range: Tuple[int, int] = (-2, 2),
        # Noise
        background_noise_prob: float = 0.5,
        noise_snr_range: Tuple[float, float] = (5.0, 20.0),
        # RIR
        rir_prob: float = 0.25,
        rir_dry_wet_min: float = 0.3,
        rir_dry_wet_max: float = 0.7,
        # Background noise and RIR paths
        background_noise_files: Optional[List[Path]] = None,
        rir_files: Optional[List[Path]] = None
    ):
        """
        Initialize augmentation pipeline

        Args:
            sample_rate: Audio sample rate
            device: Device for computation (always 'cpu' for dataset pipeline)
            time_stretch_range: Min and max time stretch factors
            pitch_shift_range: Min and max pitch shift in semitones
            background_noise_prob: Probability of adding background noise
            noise_snr_range: SNR range in dB for noise mixing
            rir_prob: Probability of applying RIR
            rir_dry_wet_min: Minimum dry signal ratio (0.0=full wet, 1.0=full dry)
            rir_dry_wet_max: Maximum dry signal ratio
            background_noise_files: List of background noise file paths
            rir_files: List of RIR file paths
        """
        self.sample_rate = sample_rate
        self.device = 'cpu'  # Always CPU for audio augmentation

        # Time domain parameters
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_range = pitch_shift_range

        # Noise parameters
        self.background_noise_prob = background_noise_prob
        self.noise_snr_range = noise_snr_range

        # RIR parameters
        self.rir_prob = rir_prob
        self.rir_dry_wet_min = rir_dry_wet_min
        self.rir_dry_wet_max = rir_dry_wet_max

        # Preload background noise and RIRs
        self.background_noises = []
        self.rirs = []

        if background_noise_files:
            self._load_background_noises(background_noise_files)

        if rir_files:
            self._load_rirs(rir_files)

        logger.info(
            f"Augmentation initialized: {len(self.background_noises)} background noises, "
            f"{len(self.rirs)} RIRs"
        )

    def _load_background_noises(self, noise_files: List[Path]):
        """Load background noise files into memory"""
        logger.info(f"Loading {len(noise_files)} background noise files...")

        for noise_file in noise_files[:100]:  # Limit to 100 files to save memory
            try:
                waveform, sr = torchaudio.load(str(noise_file))

                # Resample if needed
                if sr != self.sample_rate:
                    resampler = T.Resample(sr, self.sample_rate)
                    waveform = resampler(waveform)

                # Convert to mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                # Move to device
                waveform = waveform.to(self.device)

                self.background_noises.append(waveform)

            except Exception as e:
                logger.warning(f"Failed to load background noise {noise_file}: {e}")

    def _validate_rir(self, waveform: torch.Tensor, file_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate RIR quality

        Args:
            waveform: RIR waveform tensor
            file_path: Path to RIR file (for logging)

        Returns:
            (is_valid, warning_message)
        """
        warnings = []

        # Duration check
        duration = waveform.shape[-1] / self.sample_rate
        if duration < 0.1 or duration > 5.0:
            return False, f"Invalid duration: {duration:.2f}s (expected 0.1-5.0s)"

        # Energy check
        energy = torch.sum(waveform ** 2).item()
        if energy < 1e-6:
            return False, "RIR has near-zero energy (silent)"

        # NaN/Inf check
        if not torch.isfinite(waveform).all():
            return False, "RIR contains NaN or Inf values"

        # Peak location check (first 10%)
        peak_idx = torch.argmax(torch.abs(waveform)).item()
        peak_position = peak_idx / waveform.shape[-1]
        if peak_position > 0.1:
            warnings.append(f"Peak at {peak_position*100:.1f}% (expected <10%)")

        warning_msg = "; ".join(warnings) if warnings else None
        return True, warning_msg

    def _load_rirs(self, rir_files: List[Path]):
        """Load RIR files into memory with validation"""
        all_rir_files = list(set(rir_files))
        max_rirs = min(len(all_rir_files), 200)

        logger.info(f"Loading up to {max_rirs} RIRs (found {len(all_rir_files)})...")

        valid_count = 0
        warning_count = 0

        for rir_file in all_rir_files[:max_rirs]:
            try:
                waveform, sr = torchaudio.load(str(rir_file))

                # Resample if needed
                if sr != self.sample_rate:
                    resampler = T.Resample(sr, self.sample_rate)
                    waveform = resampler(waveform)

                # Convert to mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                # Validate RIR quality
                is_valid, warning_msg = self._validate_rir(waveform, rir_file)

                if not is_valid:
                    logger.warning(f"Skipping invalid RIR {rir_file.name}: {warning_msg}")
                    continue

                if warning_msg:
                    logger.debug(f"RIR quality warning for {rir_file.name}: {warning_msg}")
                    warning_count += 1

                # Normalize
                waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)

                # Move to device
                waveform = waveform.to(self.device)

                self.rirs.append(waveform)
                valid_count += 1

            except Exception as e:
                logger.warning(f"Failed to load RIR {rir_file}: {e}")

        logger.info(f"Loaded {valid_count} valid RIRs ({warning_count} with quality warnings)")

    def time_stretch(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply time stretching (speed change without pitch change)

        Args:
            waveform: Input waveform tensor (channels, samples)

        Returns:
            Time-stretched waveform
        """
        stretch_factor = random.uniform(*self.time_stretch_range)

        # Simple time stretching using resampling
        # Note: This is a simplified version. For better quality, use librosa
        original_length = waveform.shape[-1]
        new_length = int(original_length / stretch_factor)

        # Resample to new length
        stretched = torch.nn.functional.interpolate(
            waveform.unsqueeze(0),
            size=new_length,
            mode='linear',
            align_corners=False
        ).squeeze(0)

        # Pad or trim to original length
        if stretched.shape[-1] < original_length:
            padding = original_length - stretched.shape[-1]
            stretched = torch.nn.functional.pad(stretched, (0, padding))
        else:
            stretched = stretched[:, :original_length]

        return stretched

    def pitch_shift(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply pitch shifting

        Args:
            waveform: Input waveform tensor (channels, samples)

        Returns:
            Pitch-shifted waveform
        """
        n_steps = random.randint(*self.pitch_shift_range)

        if n_steps == 0:
            return waveform

        # Calculate shift rate
        shift_rate = 2 ** (n_steps / 12)

        # Pitch shift using time stretch and resample
        # Stretch by inverse of shift rate
        stretched = torch.nn.functional.interpolate(
            waveform.unsqueeze(0),
            size=int(waveform.shape[-1] / shift_rate),
            mode='linear',
            align_corners=False
        ).squeeze(0)

        # Resample back to original rate
        pitch_shifted = torch.nn.functional.interpolate(
            stretched.unsqueeze(0),
            size=waveform.shape[-1],
            mode='linear',
            align_corners=False
        ).squeeze(0)

        return pitch_shifted

    def add_background_noise(
        self,
        waveform: torch.Tensor,
        snr_db: Optional[float] = None
    ) -> torch.Tensor:
        """
        Add background noise to audio

        Args:
            waveform: Input waveform tensor (channels, samples)
            snr_db: Signal-to-noise ratio in dB (if None, random from range)

        Returns:
            Waveform with added noise
        """
        if not self.background_noises:
            # Generate white noise if no background noises loaded
            noise = torch.randn_like(waveform) * 0.01
        else:
            # Select random background noise
            noise = random.choice(self.background_noises)

            # Random segment if noise is longer than waveform
            if noise.shape[-1] > waveform.shape[-1]:
                start = random.randint(0, noise.shape[-1] - waveform.shape[-1])
                noise = noise[:, start:start + waveform.shape[-1]]
            else:
                # Repeat if noise is shorter
                repeats = (waveform.shape[-1] // noise.shape[-1]) + 1
                noise = noise.repeat(1, repeats)
                noise = noise[:, :waveform.shape[-1]]

        # Ensure same shape
        if noise.shape[0] != waveform.shape[0]:
            noise = noise.repeat(waveform.shape[0], 1)

        # Calculate SNR
        if snr_db is None:
            snr_db = random.uniform(*self.noise_snr_range)

        # Calculate signal and noise power
        signal_power = torch.mean(waveform ** 2)
        noise_power = torch.mean(noise ** 2)

        # Calculate scaling factor for desired SNR
        snr_linear = 10 ** (snr_db / 10)
        scale = torch.sqrt(signal_power / (noise_power * snr_linear + 1e-8))

        # Add scaled noise
        noisy_waveform = waveform + scale * noise

        # Normalize to prevent clipping
        max_val = torch.max(torch.abs(noisy_waveform))
        if max_val > 1.0:
            noisy_waveform = noisy_waveform / max_val

        return noisy_waveform

    def apply_rir(self, waveform: torch.Tensor, dry_wet_ratio: Optional[float] = None) -> torch.Tensor:
        """
        Apply Room Impulse Response (RIR) with dry/wet mixing

        Args:
            waveform: Input waveform tensor (channels, samples)
            dry_wet_ratio: Dry signal ratio (0.0=full wet, 1.0=full dry)
                          If None, random value from config range

        Returns:
            Mixed waveform (dry + wet)
        """
        if not self.rirs:
            return waveform

        # Store original (dry) waveform
        dry_signal = waveform.clone()

        # Select random RIR
        rir = random.choice(self.rirs)

        # Store original energy
        original_energy = torch.mean(waveform ** 2)

        # Convolve with RIR to create wet signal
        # Convert to 1D for convolution
        waveform_1d = waveform.squeeze(0)
        rir_1d = rir.squeeze(0)

        # Perform convolution
        wet_signal = torch.nn.functional.conv1d(
            waveform_1d.unsqueeze(0).unsqueeze(0),
            rir_1d.unsqueeze(0).unsqueeze(0),
            padding=rir_1d.shape[0] // 2
        )

        # Trim to original length
        wet_signal = wet_signal.squeeze(0)[:, :waveform.shape[-1]]

        # Remove DC offset
        wet_signal = wet_signal - torch.mean(wet_signal)

        # Energy normalization: restore original energy to wet signal
        wet_energy = torch.mean(wet_signal ** 2)
        if wet_energy > 1e-8:
            wet_signal = wet_signal * torch.sqrt(original_energy / (wet_energy + 1e-8))

        # Determine dry/wet ratio
        if dry_wet_ratio is None:
            dry_wet_ratio = random.uniform(self.rir_dry_wet_min, self.rir_dry_wet_max)

        # Mix dry and wet signals
        # dry_ratio: portion of dry signal (0.0 to 1.0)
        # wet_ratio: portion of wet signal (1.0 - dry_ratio)
        mixed = dry_wet_ratio * dry_signal + (1.0 - dry_wet_ratio) * wet_signal

        # Final normalization if needed
        max_val = torch.max(torch.abs(mixed))
        if max_val > 1.0:
            mixed = mixed / max_val

        return mixed

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation pipeline on CPU

        Args:
            waveform: Input waveform tensor (channels, samples)

        Returns:
            Augmented waveform
        """
        # Ensure waveform is on CPU
        if waveform.device.type != 'cpu':
            waveform = waveform.cpu()

        # Apply augmentations randomly

        # Time stretch
        if random.random() < 0.5:
            waveform = self.time_stretch(waveform)

        # Pitch shift
        if random.random() < 0.5:
            waveform = self.pitch_shift(waveform)

        # Background noise
        if random.random() < self.background_noise_prob:
            waveform = self.add_background_noise(waveform)

        # RIR
        if random.random() < self.rir_prob:
            waveform = self.apply_rir(waveform)

        return waveform


class SpecAugment:
    """
    SpecAugment for spectrograms (frequency and time masking)
    """

    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
        n_freq_masks: int = 2,
        n_time_masks: int = 2
    ):
        """
        Initialize SpecAugment

        Args:
            freq_mask_param: Maximum width of frequency mask
            time_mask_param: Maximum width of time mask
            n_freq_masks: Number of frequency masks to apply
            n_time_masks: Number of time masks to apply
        """
        self.freq_mask = T.FrequencyMasking(freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param)
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment

        Args:
            spectrogram: Input spectrogram tensor (channels, freq, time)

        Returns:
            Augmented spectrogram
        """
        # Apply frequency masking
        for _ in range(self.n_freq_masks):
            spectrogram = self.freq_mask(spectrogram)

        # Apply time masking
        for _ in range(self.n_time_masks):
            spectrogram = self.time_mask(spectrogram)

        return spectrogram


if __name__ == "__main__":
    # Test augmentation
    print("Audio Augmentation Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create augmentation
    augmentation = AudioAugmentation(
        sample_rate=16000,
        device=device,
        time_stretch_range=(0.9, 1.1),
        pitch_shift_range=(-2, 2),
        background_noise_prob=0.5,
        noise_snr_range=(5.0, 20.0),
        rir_prob=0.25
    )

    # Test with random audio
    test_audio = torch.randn(1, 16000).to(device)  # 1 second of audio
    print(f"Input shape: {test_audio.shape}")

    # Apply augmentation
    augmented = augmentation(test_audio)
    print(f"Output shape: {augmented.shape}")

    print("\n✅ Augmentation test complete")
    print("Augmentation module loaded successfully")