"""
Unit Tests for Model Architectures
Tests all model architectures for correct output shapes and basic functionality
"""
import pytest
import torch
import torch.nn as nn


class TestModelArchitectures:
    """Test suite for model architectures"""

    @pytest.mark.unit
    def test_resnet18_forward_shape(self, resnet_model, sample_spectrogram, device):
        """Test ResNet18 produces correct output shape"""
        model = resnet_model.to(device)
        inputs = sample_spectrogram.to(device)
        
        outputs = model(inputs)
        
        assert outputs.shape == (2, 2), f"Expected (2, 2), got {outputs.shape}"
    
    @pytest.mark.unit
    def test_resnet18_embed_shape(self, resnet_model, sample_spectrogram, device):
        """Test ResNet18 embedding extraction"""
        model = resnet_model.to(device)
        inputs = sample_spectrogram.to(device)
        
        embeddings = model.embed(inputs)
        
        assert embeddings.dim() == 2, "Embeddings should be 2D"
        assert embeddings.shape[0] == 2, "Batch dimension should be preserved"

    @pytest.mark.unit
    def test_mobilenetv3_forward_shape(self, mobilenet_model, sample_spectrogram, device):
        """Test MobileNetV3 produces correct output shape"""
        model = mobilenet_model.to(device)
        inputs = sample_spectrogram.to(device)
        
        outputs = model(inputs)
        
        assert outputs.shape == (2, 2), f"Expected (2, 2), got {outputs.shape}"

    @pytest.mark.unit
    def test_lstm_forward_shape(self, lstm_model, device):
        """Test LSTM produces correct output shape"""
        model = lstm_model.to(device)
        # LSTM expects (batch, time_steps, features)
        inputs = torch.randn(2, 50, 40).to(device)
        
        outputs = model(inputs)
        
        assert outputs.shape == (2, 2), f"Expected (2, 2), got {outputs.shape}"

    @pytest.mark.unit
    def test_gru_forward_shape(self, device):
        """Test GRU produces correct output shape"""
        from src.models.architectures import create_model
        
        model = create_model("gru", num_classes=2, input_size=40).to(device)
        inputs = torch.randn(2, 50, 40).to(device)
        
        outputs = model(inputs)
        
        assert outputs.shape == (2, 2), f"Expected (2, 2), got {outputs.shape}"

    @pytest.mark.unit
    def test_tcn_forward_shape(self, device):
        """Test TCN produces correct output shape"""
        from src.models.architectures import create_model
        
        model = create_model("tcn", num_classes=2, input_size=40).to(device)
        inputs = torch.randn(2, 50, 40).to(device)
        
        outputs = model(inputs)
        
        assert outputs.shape == (2, 2), f"Expected (2, 2), got {outputs.shape}"

    @pytest.mark.unit
    def test_tiny_conv_forward_shape(self, device):
        """Test TinyConv produces correct output shape"""
        from src.models.architectures import create_model
        
        model = create_model("tiny_conv", num_classes=2).to(device)
        inputs = torch.randn(2, 1, 64, 50).to(device)
        
        outputs = model(inputs)
        
        assert outputs.shape == (2, 2), f"Expected (2, 2), got {outputs.shape}"

    @pytest.mark.unit
    def test_invalid_architecture_raises(self):
        """Test that invalid architecture name raises ValueError"""
        from src.models.architectures import create_model
        
        with pytest.raises(ValueError, match="Unknown architecture"):
            create_model("invalid_model", num_classes=2)

    @pytest.mark.unit
    def test_model_gradient_flow(self, resnet_model, sample_spectrogram, sample_labels, device):
        """Test gradients flow through model"""
        model = resnet_model.to(device)
        inputs = sample_spectrogram.to(device)
        labels = sample_labels.to(device)
        
        model.train()
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        
        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestModelExport:
    """Test model export functionality"""

    @pytest.mark.unit
    def test_model_to_eval_mode(self, resnet_model):
        """Test model can switch to eval mode"""
        resnet_model.train()
        assert resnet_model.training is True
        
        resnet_model.eval()
        assert resnet_model.training is False

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_model_channels_last(self, resnet_model, sample_spectrogram):
        """Test model works with channels_last memory format"""
        device = "cuda"
        model = resnet_model.to(device, memory_format=torch.channels_last)
        inputs = sample_spectrogram.to(device, memory_format=torch.channels_last)
        
        outputs = model(inputs)
        
        assert outputs.shape == (2, 2)
