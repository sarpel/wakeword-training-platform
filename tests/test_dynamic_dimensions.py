import pytest
import torch

from src.config.defaults import WakewordConfig
from src.data.feature_extraction import FeatureExtractor
from src.data.processor import AudioProcessor
from src.models.architectures import create_model


def test_feature_extractor_dynamic_dimensions():
    """Verify FeatureExtractor handles different mel dimensions correctly"""
    for n_mels in [40, 64, 80, 128]:
        extractor = FeatureExtractor(n_mels=n_mels)
        dummy_audio = torch.randn(1, 16000)
        features = extractor(dummy_audio)

        # Output shape: (batch, channels, n_mels, time)
        assert features.shape[2] == n_mels
        print(f"✅ FeatureExtractor verified for n_mels={n_mels}")


def test_model_input_dynamic_dimensions():
    """Verify models can be created with various input sizes derived from n_mels"""
    architectures = ["resnet18", "mobilenetv3", "lstm", "gru", "tcn", "tiny_conv", "conformer"]

    for n_mels in [40, 64, 80]:
        # Calculate input size as it would be in the real pipeline
        # (Simplified calculation for testing)
        input_size = n_mels

        for arch in architectures:
            try:
                if arch in ["resnet18", "mobilenetv3", "tiny_conv"]:
                    model = create_model(arch, input_channels=1)
                    test_input = torch.randn(2, 1, n_mels, 50)
                else:
                    model = create_model(arch, input_size=input_size)
                    test_input = torch.randn(2, 50, n_mels)

                output = model(test_input)
                assert output.shape[1] == 2  # num_classes
                print(f"✅ Model {arch} verified for n_mels={n_mels}")
            except Exception as e:
                pytest.fail(f"Failed to create/run model {arch} with n_mels={n_mels}: {e}")


def test_audio_processor_mismatch_flag():
    """Verify AudioProcessor sets mismatch flag correctly"""
    import json
    import os

    from src.config.paths import paths

    # Create a dummy CMVN stats file with 40 dimensions
    dummy_stats_path = paths.DATA / "test_mismatch_cmvn.json"
    stats_dict = {"mean": [0.0] * 40, "std": [1.0] * 40, "count": 1000, "eps": 1e-8}
    with open(dummy_stats_path, "w") as f:
        json.dump(stats_dict, f)

    try:
        # Config with 64 dimensions
        config = WakewordConfig()
        config.data.n_mels = 64

        # Processor should detect mismatch
        processor = AudioProcessor(config, cmvn_path=dummy_stats_path, device="cpu")
        assert processor.cmvn_mismatch is True
        assert processor.cmvn is None
        print("✅ AudioProcessor correctly detected CMVN dimension mismatch")

        # Config with 40 dimensions
        config.data.n_mels = 40
        processor = AudioProcessor(config, cmvn_path=dummy_stats_path, device="cpu")
        assert processor.cmvn_mismatch is False
        assert processor.cmvn is not None
        print("✅ AudioProcessor correctly matched CMVN dimensions")

    finally:
        if dummy_stats_path.exists():
            dummy_stats_path.unlink()


if __name__ == "__main__":
    test_feature_extractor_dynamic_dimensions()
    test_model_input_dynamic_dimensions()
    test_audio_processor_mismatch_flag()
