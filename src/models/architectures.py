"""
Model Architectures for Wakeword Detection
Supports: ResNet18, MobileNetV3, LSTM, GRU, TCN
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ResNet18Wakeword(nn.Module):
    """ResNet18 adapted for wakeword detection"""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = False,
        dropout: float = 0.3,
        input_channels: int = 1
    ):
        """
        Initialize ResNet18 for wakeword detection

        Args:
            num_classes: Number of output classes
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate
            input_channels: Number of input channels (1 for mono spectrogram)
        """
        super().__init__()

        # Load ResNet18
        if pretrained:
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.resnet = models.resnet18(weights=None)

        # Modify first conv layer for single channel input if needed
        if input_channels != 3:
            self.resnet.conv1 = nn.Conv2d(
                input_channels, 64,
                kernel_size=7, stride=2, padding=3, bias=False
            )

        # Replace final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch, channels, height, width)

        Returns:
            Output logits (batch, num_classes)
        """
        return self.resnet(x)


class MobileNetV3Wakeword(nn.Module):
    """MobileNetV3-Small adapted for wakeword detection"""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = False,
        dropout: float = 0.3,
        input_channels: int = 1
    ):
        """
        Initialize MobileNetV3 for wakeword detection

        Args:
            num_classes: Number of output classes
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate
            input_channels: Number of input channels
        """
        super().__init__()

        # Load MobileNetV3-Small
        if pretrained:
            self.mobilenet = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            )
        else:
            self.mobilenet = models.mobilenet_v3_small(weights=None)

        # Modify first conv layer for single channel input if needed
        if input_channels != 3:
            self.mobilenet.features[0][0] = nn.Conv2d(
                input_channels, 16,
                kernel_size=3, stride=2, padding=1, bias=False
            )

        # Replace classifier
        num_features = self.mobilenet.classifier[0].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch, channels, height, width)

        Returns:
            Output logits (batch, num_classes)
        """
        return self.mobilenet(x)


class LSTMWakeword(nn.Module):
    """LSTM-based wakeword detector"""

    def __init__(
        self,
        input_size: int = 40,  # n_mfcc
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3
    ):
        """
        Initialize LSTM for wakeword detection

        Args:
            input_size: Input feature dimension
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            bidirectional: Use bidirectional LSTM
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Output layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch, time_steps, features)

        Returns:
            Output logits (batch, num_classes)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use final hidden state
        if self.bidirectional:
            # Concatenate forward and backward final states
            h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_n = h_n[-1]

        # Classification
        output = self.fc(h_n)

        return output


class GRUWakeword(nn.Module):
    """GRU-based wakeword detector"""

    def __init__(
        self,
        input_size: int = 40,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3
    ):
        """
        Initialize GRU for wakeword detection

        Args:
            input_size: Input feature dimension
            hidden_size: GRU hidden size
            num_layers: Number of GRU layers
            num_classes: Number of output classes
            bidirectional: Use bidirectional GRU
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Output layer
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_output_size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch, time_steps, features)

        Returns:
            Output logits (batch, num_classes)
        """
        # GRU forward
        gru_out, h_n = self.gru(x)

        # Use final hidden state
        if self.bidirectional:
            # Concatenate forward and backward final states
            h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_n = h_n[-1]

        # Classification
        output = self.fc(h_n)

        return output


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network (TCN)"""

    def __init__(
        self,
        input_channels: int = 40,
        num_channels: list = [64, 128, 256],
        kernel_size: int = 3,
        dropout: float = 0.3
    ):
        """
        Initialize TCN

        Args:
            input_channels: Number of input channels
            num_channels: List of channel sizes for each layer
            kernel_size: Convolution kernel size
            dropout: Dropout rate
        """
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_channels if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels, out_channels,
                    kernel_size, stride=1, dilation=dilation,
                    padding=(kernel_size - 1) * dilation,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch, channels, length)

        Returns:
            Output tensor (batch, channels, length)
        """
        return self.network(x)


class TemporalBlock(nn.Module):
    """Temporal block for TCN"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.3
    ):
        """Initialize temporal block"""
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Downsample for residual connection if needed
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Trim to match input size
        out = out[:, :, :x.size(2)]

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNWakeword(nn.Module):
    """TCN-based wakeword detector"""

    def __init__(
        self,
        input_size: int = 40,
        num_channels: list = [64, 128, 256],
        kernel_size: int = 3,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize TCN for wakeword detection

        Args:
            input_size: Input feature dimension
            num_channels: List of channel sizes
            kernel_size: Convolution kernel size
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()

        self.tcn = TemporalConvNet(
            input_channels=input_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )

        # Global average pooling and classifier
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1], num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch, time_steps, features) or (batch, features, time_steps)

        Returns:
            Output logits (batch, num_classes)
        """
        # TCN expects (batch, channels, length)
        if x.dim() == 3 and x.size(1) < x.size(2):
            # Assume (batch, time, features) -> transpose to (batch, features, time)
            x = x.transpose(1, 2)

        # TCN forward
        tcn_out = self.tcn(x)

        # Classification
        output = self.fc(tcn_out)

        return output


def create_model(
    architecture: str,
    num_classes: int = 2,
    pretrained: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models

    Args:
        architecture: Model architecture name
        num_classes: Number of output classes
        pretrained: Use pretrained weights (for ResNet/MobileNet)
        **kwargs: Additional model-specific arguments

    Returns:
        PyTorch model

    Raises:
        ValueError: If architecture is not recognized
    """
    architecture = architecture.lower()

    if architecture == "resnet18":
        return ResNet18Wakeword(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=kwargs.get('dropout', 0.3),
            input_channels=kwargs.get('input_channels', 1)
        )

    elif architecture == "mobilenetv3":
        return MobileNetV3Wakeword(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=kwargs.get('dropout', 0.3),
            input_channels=kwargs.get('input_channels', 1)
        )

    elif architecture == "lstm":
        return LSTMWakeword(
            input_size=kwargs.get('input_size', 40),
            hidden_size=kwargs.get('hidden_size', 128),
            num_layers=kwargs.get('num_layers', 2),
            num_classes=num_classes,
            bidirectional=kwargs.get('bidirectional', True),
            dropout=kwargs.get('dropout', 0.3)
        )

    elif architecture == "gru":
        return GRUWakeword(
            input_size=kwargs.get('input_size', 40),
            hidden_size=kwargs.get('hidden_size', 128),
            num_layers=kwargs.get('num_layers', 2),
            num_classes=num_classes,
            bidirectional=kwargs.get('bidirectional', True),
            dropout=kwargs.get('dropout', 0.3)
        )

    elif architecture == "tcn":
        return TCNWakeword(
            input_size=kwargs.get('input_size', 40),
            num_channels=kwargs.get('num_channels', [64, 128, 256]),
            kernel_size=kwargs.get('kernel_size', 3),
            num_classes=num_classes,
            dropout=kwargs.get('dropout', 0.3)
        )

    else:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Supported: resnet18, mobilenetv3, lstm, gru, tcn"
        )


if __name__ == "__main__":
    # Test model creation
    print("Model Architectures Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Test each architecture
    architectures = ["resnet18", "mobilenetv3", "lstm", "gru", "tcn"]

    for arch in architectures:
        print(f"\nTesting {arch}...")

        model = create_model(arch, num_classes=2, pretrained=False)
        model = model.to(device)

        # Test forward pass
        if arch in ["resnet18", "mobilenetv3"]:
            # 2D input (batch, channels, height, width)
            test_input = torch.randn(2, 1, 64, 50).to(device)
        else:
            # Sequential input (batch, time, features)
            test_input = torch.randn(2, 50, 40).to(device)

        output = model(test_input)
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  ✅ {arch} works correctly")

    print("\n✅ All architectures tested successfully")
    print("Model architectures module loaded successfully")