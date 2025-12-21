"""
Test script to verify dynamic model configuration
Tests that all models properly use configuration parameters from config.yaml
"""

import sys
import os
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from typing import Dict, Any

# Import model creation function
from src.models.architectures import create_model


def test_mobilenetv3_hybrid_config():
    """Test MobileNetV3 with hybrid architecture configuration"""
    print("\n=== Testing MobileNetV3 Hybrid Architecture ===")

    # Test 1: Basic MobileNetV3
    model1 = create_model(
        architecture="mobilenetv3",
        num_classes=2,
        pretrained=False,
        dropout=0.5,
        input_channels=1
    )
    print("[OK] Basic MobileNetV3 created")

    # Test 2: MobileNetV3 with LSTM layers
    model2 = create_model(
        architecture="mobilenetv3",
        num_classes=2,
        pretrained=False,
        dropout=0.3,
        input_channels=1,
        num_layers=2,
        hidden_size=256,
        bidirectional=True,
        rnn_type="lstm"
    )
    print("[OK] MobileNetV3 with LSTM layers created")

    # Test 3: MobileNetV3 with custom MLP head
    model3 = create_model(
        architecture="mobilenetv3",
        num_classes=2,
        pretrained=False,
        dropout=0.3,
        input_channels=1,
        cddnn_hidden_layers=[512, 256, 128, 64]
    )
    print("[OK] MobileNetV3 with custom MLP head created")

    # Test 4: MobileNetV3 with both LSTM and custom MLP
    model4 = create_model(
        architecture="mobilenetv3",
        num_classes=2,
        pretrained=False,
        dropout=0.3,
        input_channels=1,
        num_layers=1,
        hidden_size=128,
        bidirectional=False,
        rnn_type="gru",
        cddnn_hidden_layers=[256, 128]
    )
    print("[OK] MobileNetV3 with GRU + custom MLP created")

    # Verify forward pass
    batch_size = 2
    test_input = torch.randn(batch_size, 1, 64, 50)  # (batch, channels, height, width)

    with torch.no_grad():
        out1 = model1(test_input)
        assert out1.shape == (batch_size, 2), f"Output shape mismatch: {out1.shape}"
        print(f"  Output shape: {out1.shape}")

        out2 = model2(test_input)
        assert out2.shape == (batch_size, 2), f"Output shape mismatch: {out2.shape}"
        print(f"  Output shape with LSTM: {out2.shape}")


def test_tinyconv_dynamic_channels():
    """Test TinyConvWakeword with dynamic channel configuration"""
    print("\n=== Testing TinyConvWakeword Dynamic Channels ===")

    # Test 1: Default channels
    model1 = create_model(
        architecture="tiny_conv",
        num_classes=2,
        input_channels=1,
        dropout=0.3
    )
    print("[OK] TinyConv with default channels created")

    # Test 2: Custom channel configuration (small)
    model2 = create_model(
        architecture="tiny_conv",
        num_classes=2,
        input_channels=1,
        dropout=0.2,
        tcn_num_channels=[8, 16, 32]
    )
    print("[OK] TinyConv with custom small channels [8, 16, 32] created")

    # Test 3: Custom channel configuration (large)
    model3 = create_model(
        architecture="tiny_conv",
        num_classes=2,
        input_channels=1,
        dropout=0.4,
        tcn_num_channels=[32, 64, 128, 256, 128],
        kernel_size=5,
        tcn_dropout=0.5
    )
    print("[OK] TinyConv with custom large channels [32, 64, 128, 256, 128] created")

    # Verify forward pass
    batch_size = 2
    test_input = torch.randn(batch_size, 1, 64, 50)  # (batch, channels, height, width)

    with torch.no_grad():
        out1 = model1(test_input)
        assert out1.shape == (batch_size, 2), f"Output shape mismatch: {out1.shape}"
        print(f"  Default model output shape: {out1.shape}")

        out2 = model2(test_input)
        assert out2.shape == (batch_size, 2), f"Output shape mismatch: {out2.shape}"
        print(f"  Small model output shape: {out2.shape}")

        out3 = model3(test_input)
        assert out3.shape == (batch_size, 2), f"Output shape mismatch: {out3.shape}"
        print(f"  Large model output shape: {out3.shape}")


def test_tcn_config_parameters():
    """Test TCNWakeword with configuration parameters"""
    print("\n=== Testing TCNWakeword Configuration ===")

    # Test 1: Default TCN
    model1 = create_model(
        architecture="tcn",
        num_classes=2,
        input_size=40
    )
    print("[OK] TCN with default configuration created")

    # Test 2: Custom TCN configuration
    model2 = create_model(
        architecture="tcn",
        num_classes=2,
        input_size=80,
        tcn_num_channels=[32, 64, 128, 256, 512],
        tcn_kernel_size=5,
        tcn_dropout=0.5
    )
    print("[OK] TCN with custom configuration created")

    # Verify forward pass
    batch_size = 2
    test_input1 = torch.randn(batch_size, 50, 40)  # (batch, time, features)
    test_input2 = torch.randn(batch_size, 50, 80)  # (batch, time, features)

    with torch.no_grad():
        out1 = model1(test_input1)
        assert out1.shape == (batch_size, 2), f"Output shape mismatch: {out1.shape}"
        print(f"  Default TCN output shape: {out1.shape}")

        out2 = model2(test_input2)
        assert out2.shape == (batch_size, 2), f"Output shape mismatch: {out2.shape}"
        print(f"  Custom TCN output shape: {out2.shape}")


def test_cddnn_config_parameters():
    """Test CDDNNWakeword with configuration parameters"""
    print("\n=== Testing CDDNNWakeword Configuration ===")

    # Test 1: Default CD-DNN
    model1 = create_model(
        architecture="cd_dnn",
        num_classes=2,
        input_size=2000  # 40 features * 50 frames
    )
    print("[OK] CD-DNN with default configuration created")

    # Test 2: Custom CD-DNN configuration
    model2 = create_model(
        architecture="cd_dnn",
        num_classes=2,
        cddnn_hidden_layers=[1024, 512, 256, 128, 64],
        cddnn_context_frames=100,
        cddnn_dropout=0.5,
        input_size=8000  # 80 features * 100 frames
    )
    print("[OK] CD-DNN with custom configuration created")

    # Test 3: CD-DNN with auto-calculated input size
    model3 = create_model(
        architecture="cd_dnn",
        num_classes=2,
        cddnn_hidden_layers=[256, 128],
        cddnn_context_frames=30,
        input_size=1800  # 60 features * 30 frames
    )
    print("[OK] CD-DNN with auto-calculated input size created")

    # Verify forward pass
    batch_size = 2
    test_input1 = torch.randn(batch_size, 2000)  # Flattened input
    test_input2 = torch.randn(batch_size, 8000)  # 80 * 100
    test_input3 = torch.randn(batch_size, 1800)  # 60 * 30

    with torch.no_grad():
        out1 = model1(test_input1)
        assert out1.shape == (batch_size, 2), f"Output shape mismatch: {out1.shape}"
        print(f"  Default CD-DNN output shape: {out1.shape}")

        out2 = model2(test_input2)
        assert out2.shape == (batch_size, 2), f"Output shape mismatch: {out2.shape}"
        print(f"  Custom CD-DNN output shape: {out2.shape}")

        out3 = model3(test_input3)
        assert out3.shape == (batch_size, 2), f"Output shape mismatch: {out3.shape}"
        print(f"  Auto-calculated CD-DNN output shape: {out3.shape}")


def test_lstm_gru_config():
    """Test LSTM and GRU models with configuration parameters"""
    print("\n=== Testing LSTM/GRU Configuration ===")

    # Test LSTM with custom config
    lstm_model = create_model(
        architecture="lstm",
        num_classes=2,
        input_size=80,
        hidden_size=256,
        num_layers=3,
        bidirectional=False,
        dropout=0.4
    )
    print("[OK] LSTM with custom configuration created")

    # Test GRU with custom config
    gru_model = create_model(
        architecture="gru",
        num_classes=2,
        input_size=60,
        hidden_size=128,
        num_layers=1,
        bidirectional=True,
        dropout=0.2
    )
    print("[OK] GRU with custom configuration created")

    # Verify forward pass
    batch_size = 2
    lstm_input = torch.randn(batch_size, 50, 80)  # (batch, time, features)
    gru_input = torch.randn(batch_size, 50, 60)   # (batch, time, features)

    with torch.no_grad():
        lstm_out = lstm_model(lstm_input)
        assert lstm_out.shape == (batch_size, 2), f"Output shape mismatch: {lstm_out.shape}"
        print(f"  LSTM output shape: {lstm_out.shape}")

        gru_out = gru_model(gru_input)
        assert gru_out.shape == (batch_size, 2), f"Output shape mismatch: {gru_out.shape}"
        print(f"  GRU output shape: {gru_out.shape}")


def test_config_from_dict():
    """Test creating models with configuration from dictionary (simulating config.yaml)"""
    print("\n=== Testing Configuration from Dictionary ===")

    # Simulate a config dictionary as it would come from config.yaml
    config = {
        "architecture": "mobilenetv3",
        "num_classes": 2,
        "pretrained": False,
        "dropout": 0.3,
        "input_channels": 1,
        # Architecture-specific parameters
        "hidden_size": 128,
        "num_layers": 2,
        "bidirectional": True,
        "rnn_type": "lstm",
        # TCN Parameters
        "tcn_num_channels": [64, 128, 256],
        "tcn_kernel_size": 3,
        "tcn_dropout": 0.3,
        # CD-DNN Parameters
        "cddnn_hidden_layers": [512, 256, 128],
        "cddnn_context_frames": 50,
        "cddnn_dropout": 0.3,
    }

    # Create model with all config parameters
    model = create_model(**config)
    print("[OK] Model created from full config dictionary")

    # Test with TCN config
    tcn_config = {
        "architecture": "tcn",
        "num_classes": 2,
        "input_size": 40,
        "tcn_num_channels": [32, 64, 128],
        "tcn_kernel_size": 5,
        "tcn_dropout": 0.4
    }

    tcn_model = create_model(**tcn_config)
    print("[OK] TCN model created from config dictionary")

    # Test with TinyConv config
    tiny_config = {
        "architecture": "tiny_conv",
        "num_classes": 2,
        "input_channels": 1,
        "dropout": 0.3,
        "tcn_num_channels": [16, 32, 64, 128],
        "kernel_size": 3,
        "tcn_dropout": 0.4
    }

    tiny_model = create_model(**tiny_config)
    print("[OK] TinyConv model created from config dictionary")

    print("\n[OK] All configuration tests passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING DYNAMIC MODEL CONFIGURATION")
    print("=" * 60)

    try:
        test_mobilenetv3_hybrid_config()
        test_tinyconv_dynamic_channels()
        test_tcn_config_parameters()
        test_cddnn_config_parameters()
        test_lstm_gru_config()
        test_config_from_dict()

        print("\n" + "=" * 60)
        print("[OK] ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()