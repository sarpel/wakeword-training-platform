import logging
from pathlib import Path

import torch

from src.config.defaults import WakewordConfig
from src.evaluation.evaluator import load_model_for_evaluation
from src.models.architectures import create_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_tiny_conv_architecture_loading():
    print("\nTesting TinyConv Architecture Loading Fix...")

    # 1. Setup: Create a custom config with non-default architecture params
    config = WakewordConfig()
    config.model.architecture = "tiny_conv"
    # Use non-default channel configuration to verify it's preserved
    # Default is [16, 32, 64, 64]
    custom_channels = [8, 16, 32]
    config.model.tcn_num_channels = custom_channels
    config.data.n_mels = 40
    config.data.audio_duration = 1.0
    config.data.sample_rate = 16000

    # 2. Create the original model
    print(f"Creating original model with channels: {custom_channels}")
    original_model = create_model(
        architecture=config.model.architecture,
        num_classes=config.model.num_classes,
        input_size=config.data.n_mels,
        tcn_num_channels=custom_channels,
    )

    # Verify original model structure
    # tiny_conv uses nn.Sequential for features.
    # With 3 layers, it should have 3 Conv2d blocks.
    # Each block is Conv -> BN -> ReLU -> (Dropout)
    # Total modules: 3 * 3 = 9 (assuming no dropout for simplicity or ignoring it)
    print(f"Original model features: {original_model.features}")

    # 3. Save checkpoint
    checkpoint_path = Path("test_arch_fix.pt")
    torch.save(
        {"model_state_dict": original_model.state_dict(), "config": config.to_dict(), "epoch": 0, "val_loss": 0.0},
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")

    # 4. Load model using the evaluator's logic
    print("Loading model via evaluator...")
    try:
        # We simulate the load call.
        # Note: load_model_for_evaluation calls create_model internally using the config from checkpoint.
        loaded_model, info = load_model_for_evaluation(checkpoint_path, device="cpu")

        print("Model loaded successfully.")
        print(f"Loaded model features: {loaded_model.features}")

        # 5. Verification
        # Check if weights match exactly
        original_state = original_model.state_dict()
        loaded_state = loaded_model.state_dict()

        mismatch = False
        for key in original_state:
            if key not in loaded_state:
                print(f"❌ Key missing in loaded model: {key}")
                mismatch = True
                continue

            if not torch.equal(original_state[key], loaded_state[key]):
                print(f"❌ Weight mismatch for {key}")
                mismatch = True

        # Check for extra keys in loaded model (which would indicate it was created with default larger arch)
        for key in loaded_state:
            if key not in original_state:
                print(f"❌ Extra key in loaded model (Architectural mismatch!): {key}")
                mismatch = True

        if not mismatch:
            print("✅ PASS: Loaded model matches original architecture and weights exactly.")
        else:
            print("❌ FAIL: Model mismatch detected.")

    except Exception as e:
        print(f"❌ FAIL: Loading raised exception: {e}")
        import traceback

        traceback.print_exc()

    finally:
        if checkpoint_path.exists():
            checkpoint_path.unlink()


if __name__ == "__main__":
    test_tiny_conv_architecture_loading()
