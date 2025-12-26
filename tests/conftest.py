"""
Pytest Configuration and Fixtures
Shared fixtures for all tests
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ==============================================================================
# Hardware Fixtures
# ==============================================================================


@pytest.fixture(scope="session")
def device() -> str:
    """Get available device (CUDA preferred)"""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def cuda_available() -> bool:
    """Check if CUDA is available"""
    return torch.cuda.is_available()


# ==============================================================================
# Configuration Fixtures
# ==============================================================================


@pytest.fixture
def default_config():
    """Get default configuration"""
    from src.config.defaults import WakewordConfig

    return WakewordConfig()


@pytest.fixture
def minimal_config():
    """Minimal config for fast tests"""
    from src.config.defaults import WakewordConfig

    config = WakewordConfig()
    config.training.epochs = 2
    config.training.batch_size = 4
    config.training.num_workers = 0
    return config


# ==============================================================================
# Data Fixtures
# ==============================================================================


@pytest.fixture
def sample_audio() -> np.ndarray:
    """Generate synthetic audio sample (1.5s @ 16kHz)"""
    duration = 1.5
    sample_rate = 16000
    samples = int(duration * sample_rate)

    # Generate synthetic waveform (sine wave with noise)
    t = np.linspace(0, duration, samples)
    frequency = 440  # Hz
    audio = np.sin(2 * np.pi * frequency * t) * 0.5
    audio += np.random.randn(samples) * 0.1

    return audio.astype(np.float32)


@pytest.fixture
def sample_spectrogram() -> torch.Tensor:
    """Generate synthetic spectrogram tensor"""
    # Shape: (batch, channels, n_mels, time_steps)
    return torch.randn(2, 1, 64, 50)


@pytest.fixture
def sample_labels() -> torch.Tensor:
    """Generate binary labels"""
    return torch.tensor([0, 1], dtype=torch.long)


# ==============================================================================
# Model Fixtures
# ==============================================================================


@pytest.fixture
def resnet_model():
    """Create ResNet18 model for testing"""
    from src.models.architectures import create_model

    return create_model("resnet18", num_classes=2, pretrained=False)


@pytest.fixture
def mobilenet_model():
    """Create MobileNetV3 model for testing"""
    from src.models.architectures import create_model

    return create_model("mobilenetv3", num_classes=2, pretrained=False)


@pytest.fixture
def lstm_model():
    """Create LSTM model for testing"""
    from src.models.architectures import create_model

    return create_model("lstm", num_classes=2, input_size=40)


# ==============================================================================
# Temporary Directory Fixtures
# ==============================================================================


@pytest.fixture
def temp_data_dir(tmp_path) -> Path:
    """Create temporary data directory structure"""
    data_dir = tmp_path / "data"
    (data_dir / "raw" / "positive").mkdir(parents=True)
    (data_dir / "raw" / "negative").mkdir(parents=True)
    (data_dir / "splits").mkdir(parents=True)
    (data_dir / "npy").mkdir(parents=True)
    return data_dir


@pytest.fixture
def temp_checkpoint_dir(tmp_path) -> Path:
    """Create temporary checkpoint directory"""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    return checkpoint_dir


# ==============================================================================
# Skip Markers
# ==============================================================================


def pytest_configure(config):
    """Configure custom markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "gpu: Tests requiring CUDA GPU")


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU tests if CUDA not available"""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
