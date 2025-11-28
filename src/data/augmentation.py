"""
Audio Data Augmentation Pipeline
GPU-accelerated augmentation using torchaudio and torch operations
"""
import random
from pathlib import Path
from typing import List, Optional, Tuple

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

logger = structlog.get_logger(__name__)


class AudioAugmentation(nn.Module):
    """
    GPU-accelerated audio augmentation pipeline.
    Inherits from nn.Module for seamless integration with training pipelines.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        device: str = "cpu",  # Initial device
        # Time domain
        time_stretch_range: Tuple[float, float] = (0.9, 1.1),
        pitch_shift_range: Tuple[int, int] = (-2, 2),
        # Time shift (New)
        time_shift_prob: float = 0.5,
        time_shift_range_ms: Tuple[int, int] = (-100, 100),
        # Noise
        background_noise_prob: float = 0.5,
        noise_snr_range: Tuple[float, float] = (5.0, 20.0),
        # RIR
        rir_prob: float = 0.25,
        rir_dry_wet_min: float = 0.3,
        rir_dry_wet_max: float = 0.7,
        # Background noise and RIR paths
        background_noise_files: Optional[List[Path]] = None,
        rir_files: Optional[List[Path]] = None,
    ):
        """
        Initialize augmentation pipeline
        """
        super().__init__()
        self.sample_rate = sample_rate
        
        # Store parameters
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_range = pitch_shift_range
        self.time_shift_prob = time_shift_prob
        self.time_shift_range_ms = time_shift_range_ms
        self.background_noise_prob = background_noise_prob
        self.noise_snr_range = noise_snr_range
        self.rir_prob = rir_prob
        self.rir_dry_wet_min = rir_dry_wet_min
        self.rir_dry_wet_max = rir_dry_wet_max

        # Buffers for noises and RIRs
        # We initialize them as empty buffers. If files are provided, we load them.
        self.register_buffer("background_noises", torch.empty(0))
        self.register_buffer("rirs", torch.empty(0))
        
        if background_noise_files:
            self._load_background_noises(background_noise_files)

        if rir_files:
            self._load_rirs(rir_files)

        if device and device != "cpu":
            self.to(device)

        logger.info(
            f"Augmentation initialized: {len(self.background_noises)} background noises, "
            f"{len(self.rirs)} RIRs"
        )

    def _pad_and_stack(self, waveforms: List[torch.Tensor]) -> torch.Tensor:
        """
        Pad list of waveforms to same length and stack them.
        """
        if not waveforms:
            return torch.empty(0)
            
        max_len = max(w.shape[-1] for w in waveforms)
        padded_waveforms = []
        for w in waveforms:
            if w.shape[-1] < max_len:
                w = F.pad(w, (0, max_len - w.shape[-1]))
            padded_waveforms.append(w)
            
        # Stack: (N, 1, Samples)
        return torch.stack(padded_waveforms)

    def _load_background_noises(self, noise_files: List[Path]):
        """Load background noise files into buffer"""
        logger.info(f"Loading {len(noise_files)} background noise files...")
        loaded_noises = []

        for noise_file in noise_files[:100]:  # Limit to 100
            try:
                waveform, sr = torchaudio.load(str(noise_file))
                if sr != self.sample_rate:
                    waveform = T.Resample(sr, self.sample_rate)(waveform)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Trim to reasonable max length (e.g., 5s) to save VRAM
                max_len = self.sample_rate * 5
                if waveform.shape[-1] > max_len:
                    start = random.randint(0, waveform.shape[-1] - max_len)
                    waveform = waveform[:, start:start+max_len]
                    
                loaded_noises.append(waveform)
            except Exception as e:
                logger.warning(f"Failed to load noise {noise_file}: {e}")

        if loaded_noises:
            self.register_buffer("background_noises", self._pad_and_stack(loaded_noises))

    def _load_rirs(self, rir_files: List[Path]):
        """Load RIR files into buffer"""
        # ... similar to previous validation logic ...
        all_rir_files = list(set(rir_files))
        max_rirs = min(len(all_rir_files), 200)
        logger.info(f"Loading up to {max_rirs} RIRs...")
        
        loaded_rirs = []
        for rir_file in all_rir_files[:max_rirs]:
            try:
                waveform, sr = torchaudio.load(str(rir_file))
                if sr != self.sample_rate:
                    waveform = T.Resample(sr, self.sample_rate)(waveform)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Validate (simplified)
                energy = torch.sum(waveform**2).item()
                if energy < 1e-6 or not torch.isfinite(waveform).all():
                    continue
                    
                # Normalize
                waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
                loaded_rirs.append(waveform)
            except Exception as e:
                logger.warning(f"Failed to load RIR {rir_file}: {e}")

        if loaded_rirs:
            # BUG FIX: Use register_buffer properly by unregistering old buffer first
            stacked_rirs = self._pad_and_stack(loaded_rirs)
            # Delete existing buffer if it exists
            delattr(self, "rirs")
            self.register_buffer("rirs", stacked_rirs)

    def time_stretch(self, waveform: torch.Tensor) -> torch.Tensor:
        """Batch time stretch using interpolation"""
        # waveform: (Batch, 1, Samples)
        factor = random.uniform(*self.time_stretch_range)
        if factor == 1.0: return waveform
        
        original_len = waveform.shape[-1]
        new_len = int(original_len / factor)
        
        out = F.interpolate(waveform, size=new_len, mode="linear", align_corners=False)
        
        if new_len < original_len:
            out = F.pad(out, (0, original_len - new_len))
        else:
            out = out[..., :original_len]
            
        return out

    def pitch_shift(self, waveform: torch.Tensor) -> torch.Tensor:
        """Batch pitch shift"""
        n_steps = random.randint(*self.pitch_shift_range)
        if n_steps == 0: return waveform
        
        # Similar implementation using interpolate tricks
        shift_rate = 2 ** (n_steps / 12)
        # 1. Time stretch by 1/rate (changes pitch & speed)
        # 2. Resample back to original speed (restores speed, keeps pitch change) -> This requires Resample transform which is complex on batch variable rates
        # Simpler: Use time stretch + pretend sample rate changed (which we fix by interpolation)
        
        # Stretch
        stretched = F.interpolate(waveform, scale_factor=1/shift_rate, mode="linear", align_corners=False)
        # Resample back to original length matches original duration
        out = F.interpolate(stretched, size=waveform.shape[-1], mode="linear", align_corners=False)
        return out

    def random_time_shift(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Randomly shift audio in time (circular shift).
        Simulates streaming window misalignment.
        """
        shift_ms = random.randint(*self.time_shift_range_ms)
        if shift_ms == 0:
            return waveform
            
        shift_samples = int(shift_ms * self.sample_rate / 1000)
        return torch.roll(waveform, shifts=shift_samples, dims=-1)

    def add_background_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add background noise to batch"""
        if self.background_noises.numel() == 0:
            return waveform + torch.randn_like(waveform) * 0.01

        batch_size, _, samples = waveform.shape
        
        # Select random noises for each element in batch
        indices = torch.randint(0, len(self.background_noises), (batch_size,), device=self.background_noises.device)
        noises = self.background_noises[indices] # (B, 1, NoiseSamples)
        
        # Crop/Loop noises to match waveform length
        if noises.shape[-1] > samples:
            # Random start
            start = random.randint(0, noises.shape[-1] - samples)
            noises = noises[..., start:start+samples]
        else:
            # Repeat
            repeats = (samples // noises.shape[-1]) + 1
            noises = noises.repeat(1, 1, repeats)[..., :samples]

        # Random SNRs
        snrs = torch.empty(batch_size, 1, 1, device=waveform.device).uniform_(*self.noise_snr_range)
        snr_linear = 10 ** (snrs / 10)

        # Calculate scaling
        sig_power = waveform.pow(2).mean(dim=-1, keepdim=True)
        noise_power = noises.pow(2).mean(dim=-1, keepdim=True)
        scale = torch.sqrt(sig_power / (noise_power * snr_linear + 1e-8))
        
        noisy = waveform + scale * noises
        
        # Normalize
        max_vals = noisy.abs().max(dim=-1, keepdim=True)[0]
        noisy = noisy / (max_vals + 1e-8)
        
        return noisy

    def apply_rir(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply RIR convolution to batch"""
        if self.rirs.numel() == 0: return waveform
        
        batch_size, _, samples = waveform.shape
        
        # Select random RIRs
        indices = torch.randint(0, len(self.rirs), (batch_size,), device=self.rirs.device)
        selected_rirs = self.rirs[indices] # (B, 1, RirSamples)
        
        # Trim RIRs to remove tail silence/save compute (optional)
        
        # Convolution using FFT
        # Pad to (N + M - 1)
        n_fft = samples + selected_rirs.shape[-1] - 1
        # Next power of 2
        n_fft = 2 ** (n_fft - 1).bit_length()
        
        spec_w = torch.fft.rfft(waveform, n=n_fft)
        spec_r = torch.fft.rfft(selected_rirs, n=n_fft)
        
        convolved = torch.fft.irfft(spec_w * spec_r, n=n_fft)
        
        # Trim to original length
        wet = convolved[..., :samples]
        
        # Normalize wet
        # We match energy to input
        input_energy = waveform.pow(2).mean(dim=-1, keepdim=True)
        wet_energy = wet.pow(2).mean(dim=-1, keepdim=True)
        wet = wet * torch.sqrt(input_energy / (wet_energy + 1e-8))
        
        # Mix
        ratios = torch.empty(batch_size, 1, 1, device=waveform.device).uniform_(self.rir_dry_wet_min, self.rir_dry_wet_max)
        mixed = ratios * waveform + (1 - ratios) * wet
        
        # Final normalize
        max_vals = mixed.abs().max(dim=-1, keepdim=True)[0]
        mixed = mixed / (max_vals + 1e-8)
        
        return mixed

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations
        waveform: (Batch, 1, Samples) or (Batch, Samples) or (Samples,)
        """
        # Standardization
        original_ndim = waveform.ndim
        if original_ndim == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0) # (1, 1, S)
        elif original_ndim == 2:
            waveform = waveform.unsqueeze(1) # (B, 1, S)
            
        # Apply
        if self.training:
            if random.random() < 0.5:
                waveform = self.time_stretch(waveform)
                
            if random.random() < 0.5:
                waveform = self.pitch_shift(waveform)
                
            if random.random() < self.time_shift_prob:
                waveform = self.random_time_shift(waveform)

            if random.random() < self.background_noise_prob:
                waveform = self.add_background_noise(waveform)
                
            if random.random() < self.rir_prob:
                waveform = self.apply_rir(waveform)
                
        # Restore shape
        if original_ndim == 1:
            waveform = waveform.squeeze(0).squeeze(0)
        elif original_ndim == 2:
            waveform = waveform.squeeze(1)
            
        return waveform

# SpecAugment remains mostly the same, ensuring T.FrequencyMasking works on GPU tensors
class SpecAugment(nn.Module):
    """
    SpecAugment for spectrograms (frequency and time masking)
    """

    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
    ):
        super().__init__()
        # torchaudio.transforms.FrequencyMasking is already nn.Module
        self.freq_mask = T.FrequencyMasking(freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param)
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment
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
        rir_prob=0.25,
    )

    # Test with random audio
    test_audio = torch.randn(1, 16000).to(device)  # 1 second of audio
    print(f"Input shape: {test_audio.shape}")

    # Apply augmentation
    augmented = augmentation(test_audio)
    print(f"Output shape: {augmented.shape}")

    print("\n✅ Augmentation test complete")
    print("Augmentation module loaded successfully")