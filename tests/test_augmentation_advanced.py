from unittest.mock import MagicMock, patch

import pytest
import torch

from src.data.augmentation import AudioAugmentation


@pytest.fixture
def augmentation_cpu():
    return AudioAugmentation(
        sample_rate=16000, device="cpu", rir_prob=1.0, time_shift_prob=0.0, background_noise_prob=0.0
    )


def test_pitch_shift_basic(augmentation_cpu):
    """Test that pitch shift runs and preserves shape."""
    waveform = torch.randn(1, 1, 16000)  # 1 sec

    # Force a specific shift range to ensure execution
    augmentation_cpu.pitch_shift_range = (2, 2)

    shifted = augmentation_cpu.pitch_shift(waveform)

    assert shifted.shape == waveform.shape
    assert not torch.allclose(shifted, waveform)  # Should be different


def test_time_stretch_basic(augmentation_cpu):
    """Test that time stretch runs and preserves shape (padding/cropping)."""
    waveform = torch.randn(1, 1, 16000)

    # Force stretch
    augmentation_cpu.time_stretch_range = (1.5, 1.5)  # Speed up -> shorter -> pad
    stretched_fast = augmentation_cpu.time_stretch(waveform)
    assert stretched_fast.shape == waveform.shape

    augmentation_cpu.time_stretch_range = (0.5, 0.5)  # Slow down -> longer -> crop
    stretched_slow = augmentation_cpu.time_stretch(waveform)
    assert stretched_slow.shape == waveform.shape


def test_rir_application_no_rirs_loaded(augmentation_cpu):
    """Test RIR application when no RIRs are loaded (should be identity)."""
    waveform = torch.randn(1, 1, 16000)
    # Ensure RIRs buffer is empty
    assert len(augmentation_cpu.rirs) == 0

    out = augmentation_cpu.apply_rir(waveform)
    assert torch.allclose(out, waveform)


def test_rir_application_with_mock_rir(augmentation_cpu):
    """Test RIR application with a manually injected RIR."""
    waveform = torch.randn(1, 1, 16000)

    # Create a simple impulse RIR (delta function)
    # If we convolve with a delta at t=0, we should get the original signal (scaled by mixing)
    rir = torch.zeros(1, 1, 100)
    rir[..., 0] = 1.0

    # Inject into buffer
    augmentation_cpu.register_buffer("rirs", rir)

    # Force wet/dry to 1.0 (fully wet) for clear check, or mixed
    augmentation_cpu.rir_dry_wet_min = 1.0
    augmentation_cpu.rir_dry_wet_max = 1.0

    out = augmentation_cpu.apply_rir(waveform)

    # Since RIR is delta, output should be (close to) input, maybe normalized
    # The implementation normalizes wet signal energy to input energy
    # So it should be very close

    assert out.shape == waveform.shape
    # We check correlation or just shape/finite for now as simple convolution check
    assert torch.isfinite(out).all()


def test_forward_pass_integration(augmentation_cpu):
    """Test the full forward pass runs without error."""
    waveform = torch.randn(2, 16000)  # Batch of 2
    # Mocking random to ensure some branches are taken is hard,
    # but we can just check it returns correct shape.

    augmentation_cpu.train()  # Make sure training mode is on
    out = augmentation_cpu(waveform)
    assert out.shape == waveform.shape
