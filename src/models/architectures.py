"""
Model Architectures for Wakeword Detection
Supports: ResNet18, MobileNetV3, LSTM, GRU, TCN
"""

import logging
from typing import Any, List, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision.models as models
    from torchvision.models import quantization as quantized_models

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    models = None
    quantized_models = None

logger = logging.getLogger(__name__)


class ResNet18Wakeword(nn.Module):
    """ResNet18 adapted for wakeword detection"""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = False,
        dropout: float = 0.3,
        input_channels: int = 1,
        **kwargs: Any,  # Accept additional kwargs for flexibility
    ):
        """
        Initialize ResNet18 for wakeword detection

        Args:
            num_classes: Number of output classes
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate
            input_channels: Number of input channels (1 for mono spectrogram)
            **kwargs: Additional unused parameters for compatibility
        """
        super().__init__()

        # Load ResNet18
        if not HAS_TORCHVISION:
            raise ImportError("torchvision is required for ResNet18. Please install it.")

        # Use quantized model version if available, as it handles skip connections with FloatFunctional
        if quantized_models is not None:
            # We use quantize=False as we handle preparation manually for QAT
            self.resnet = quantized_models.resnet18(weights=None, quantize=False)
        else:
            if pretrained:
                self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.resnet = models.resnet18(weights=None)

        # Modify first conv layer for single channel input if needed
        if input_channels != 3:
            self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_features, num_classes))

        # NEW: Quantization stubs for QAT support
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch, channels, height, width)

        Returns:
            Output logits (batch, num_classes)
        """
        x = self.quant(x)
        x = self.resnet(x)
        x = self.dequant(x)
        return cast(torch.Tensor, x)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature embeddings (before classification head)
        """
        # We need to manually run the resnet layers to bypass the final fc
        # This relies on internal structure of torchvision resnet
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class MobileNetV3Wakeword(nn.Module):
    """
    MobileNetV3 architecture for wakeword detection with hybrid capabilities
    Supports adding LSTM/GRU layers and custom MLP heads based on config
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = False,
        dropout: float = 0.3,
        input_channels: int = 1,
        hidden_size: int = 128,
        num_layers: int = 0,
        bidirectional: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize MobileNetV3 for wakeword detection with hybrid architecture support

        Args:
            num_classes: Number of output classes
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate
            input_channels: Number of input channels
            hidden_size: Hidden size for RNN layers
            num_layers: Number of RNN layers (0 to disable)
            bidirectional: Whether to use bidirectional RNN
            **kwargs: Additional arguments
        """
        super().__init__()

        # Load MobileNetV3-Small
        if not HAS_TORCHVISION:
            raise ImportError("torchvision is required for MobileNetV3. Please install it.")

        if quantized_models is not None and hasattr(quantized_models, "mobilenet_v3_small"):
            self.mobilenet = quantized_models.mobilenet_v3_small(weights=None, quantize=False)
        else:
            if pretrained:
                self.mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            else:
                self.mobilenet = models.mobilenet_v3_small(weights=None)

        # Modify first conv layer for single channel input if needed
        if input_channels != 3:
            self.mobilenet.features[0][0] = nn.Conv2d(
                input_channels, 16, kernel_size=3, stride=2, padding=1, bias=False
            )

        # Extract the feature extractor (without the classifier)
        self.features = self.mobilenet.features

        # Get the number of output features from the feature extractor
        num_features = self.mobilenet.classifier[0].in_features

        # Build hybrid head based on config
        head_layers: List[nn.Module] = []

        # Add RNN layers if configured
        self.use_rnn = num_layers > 0
        if self.use_rnn:
            rnn_type = kwargs.get("rnn_type", "lstm").lower()
            # hidden_size, num_layers, bidirectional are now passed as args

            if rnn_type == "lstm":
                self.rnn: Any = nn.LSTM(
                    input_size=num_features,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    bidirectional=bidirectional,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True,
                )
            else:  # gru
                self.rnn = nn.GRU(
                    input_size=num_features,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    bidirectional=bidirectional,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True,
                )

            # Update num_features for the next layer
            num_features = hidden_size * (2 if bidirectional else 1)
        else:
            self.rnn = None

        # Build custom MLP head
        cddnn_hidden_layers = kwargs.get("cddnn_hidden_layers", [1024])

        # Build the MLP layers
        for hidden_size in cddnn_hidden_layers:
            head_layers.append(nn.Linear(num_features, hidden_size))
            head_layers.append(nn.Hardswish())
            head_layers.append(nn.Dropout(dropout))
            num_features = hidden_size

        # Add final classification layer
        head_layers.append(nn.Linear(num_features, num_classes))

        # Create the classifier head
        self.classifier = nn.Sequential(*head_layers)

        # Store whether to use adaptive pooling
        self.use_adaptive_pool = True

        # NEW: Quantization stubs for QAT support
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Output tensor of shape (batch, num_classes)
        """
        x = self.quant(x)
        # Extract features
        x = self.features(x)

        # Adaptive average pooling
        if self.use_adaptive_pool:
            x = F.adaptive_avg_pool2d(x, (1, 1))

        # Flatten
        x = x.view(x.size(0), -1)

        # Apply RNN if configured
        if self.rnn is not None:
            # Reshape for RNN: (batch, seq_len=1, features)
            x = x.unsqueeze(1)
            x, _ = self.rnn(x)
            # Take the last output
            x = x[:, -1, :]

        # Apply classifier head
        x = self.classifier(x)
        x = self.dequant(x)

        return x

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings (features before final classification layer)

        Args:
            x: Input tensor

        Returns:
            Embedding tensor
        """
        # Extract features
        x = self.features(x)

        # Adaptive average pooling
        if self.use_adaptive_pool:
            x = F.adaptive_avg_pool2d(x, (1, 1))

        # Flatten
        x = x.view(x.size(0), -1)

        # Apply RNN if configured
        if self.rnn is not None:
            x = x.unsqueeze(1)
            x, _ = self.rnn(x)
            x = x[:, -1, :]

        # Apply all layers except the last one
        for layer in self.classifier[:-1]:
            x = layer(x)

        return x


class LSTMWakeword(nn.Module):
    """LSTM-based wakeword detector"""

    def __init__(
        self,
        input_size: int = 64,  # n_mfcc
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        **kwargs: Any,  # Accept additional kwargs for flexibility
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
            **kwargs: Additional unused parameters for compatibility
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
            bidirectional=bidirectional,
        )

        # Output layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(lstm_output_size, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch, time_steps, features)

        Returns:
            Output logits (batch, num_classes)
        """
        # Input shape: (Batch, Channels, Time, Freq) or (Batch, Channels, Freq, Time)
        # RNN expects (Batch, Time, Features)

        # If 4D input (B, C, F, T)
        if x.dim() == 4:
            if x.size(1) == 1:
                x = x.squeeze(1)

        # Now x is likely 3D (B, F, T) or (B, T, F)
        if x.dim() == 3:
            # Check if we need to transpose
            # We want (B, T, F) where F is input_size
            if x.size(2) == self.lstm.input_size:
                # Already (B, T, F)
                pass
            elif x.size(1) == self.lstm.input_size:
                # Is (B, F, T) -> Transpose to (B, T, F)
                x = x.transpose(1, 2)
            else:
                # Ambiguous, assume (B, F, T) and transpose if F matches better?
                # Or just assume standard (B, F, T) coming from AudioProcessor
                if x.size(1) < x.size(2):  # Heuristic: F < T usually
                    x = x.transpose(1, 2)

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

        return cast(torch.Tensor, output)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings (final hidden state)"""
        lstm_out, (h_n, c_n) = self.lstm(x)
        if self.bidirectional:
            h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_n = h_n[-1]
        return cast(torch.Tensor, h_n)


class GRUWakeword(nn.Module):
    """GRU-based wakeword detector"""

    def __init__(
        self,
        input_size: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        **kwargs: Any,  # Accept additional kwargs for flexibility
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
            **kwargs: Additional unused parameters for compatibility
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
            bidirectional=bidirectional,
        )

        # Output layer
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(gru_output_size, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch, time_steps, features)

        Returns:
            Output logits (batch, num_classes)
        """
        # Input shape: (Batch, Channels, Time, Freq) or (Batch, Channels, Freq, Time)
        # RNN expects (Batch, Time, Features)

        # If 4D input (B, C, F, T)
        if x.dim() == 4:
            if x.size(1) == 1:
                x = x.squeeze(1)

        # Now x is likely 3D (B, F, T) or (B, T, F)
        if x.dim() == 3:
            # Check if we need to transpose
            # We want (B, T, F) where F is input_size
            if x.size(2) == self.gru.input_size:
                # Already (B, T, F)
                pass
            elif x.size(1) == self.gru.input_size:
                # Is (B, F, T) -> Transpose to (B, T, F)
                x = x.transpose(1, 2)
            else:
                # Ambiguous, assume (B, F, T) and transpose if F matches better?
                if x.size(1) < x.size(2):  # Heuristic: F < T usually
                    x = x.transpose(1, 2)

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

        return cast(torch.Tensor, output)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings (final hidden state)"""
        gru_out, h_n = self.gru(x)
        if self.bidirectional:
            h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_n = h_n[-1]
        return cast(torch.Tensor, h_n)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network (TCN)"""

    def __init__(
        self,
        input_channels: int = 64,
        num_channels: list = [64, 128, 256],
        kernel_size: int = 3,
        dropout: float = 0.3,
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
            dilation = 2**i
            in_channels = input_channels if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=(kernel_size - 1) * dilation,
                    dropout=dropout,
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
        return cast(torch.Tensor, self.network(x))


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
        dropout: float = 0.3,
    ):
        """Initialize temporal block"""
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Downsample for residual connection if needed
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Trim to match input size
        out = out[:, :, : x.size(2)]

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return cast(torch.Tensor, self.relu(self.skip_add.add(out, res)))


class TCNWakeword(nn.Module):
    """TCN-based wakeword detector"""

    def __init__(
        self,
        input_size: int = 64,
        num_channels: Optional[list] = None,
        kernel_size: int = 3,
        num_classes: int = 2,
        dropout: float = 0.3,
        **kwargs: Any,
    ):
        """
        Initialize TCN for wakeword detection

        Args:
            input_size: Input feature dimension
            num_channels: List of channel sizes (if None, uses tcn_num_channels from kwargs or defaults)
            kernel_size: Convolution kernel size (can be overridden by tcn_kernel_size in kwargs)
            num_classes: Number of output classes
            dropout: Dropout rate (can be overridden by tcn_dropout in kwargs)
            **kwargs: Additional config parameters including:
                - tcn_num_channels: List of channel sizes for TCN blocks
                - tcn_kernel_size: Kernel size for temporal convolutions
                - tcn_dropout: Dropout rate for TCN blocks
        """
        super().__init__()

        # Use config parameters from kwargs if available
        if num_channels is None:
            num_channels = kwargs.get("tcn_num_channels", [64, 128, 256])

        # Override with kwargs if present
        kernel_size = kwargs.get("tcn_kernel_size", kernel_size)
        dropout = kwargs.get("tcn_dropout", dropout)

        self.input_size = input_size  # Store input size for verification/testing
        self.tcn = TemporalConvNet(
            input_channels=input_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # Global average pooling and classifier
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch, time_steps, features) or (batch, features, time_steps)

        Returns:
            Output logits (batch, num_classes)
        """
        # Input shape: (Batch, Channels, Time, Freq) or (Batch, Channels, Freq, Time)
        # TCN expects (Batch, Channels, Length)

        # If 4D input (B, C, F, T) or (B, C, T, F)
        if x.dim() == 4:
            # Assuming (B, 1, F, T) standard from AudioProcessor
            # We want to treat Freq as channels for TCN? Or just flatten?
            # Usually TCN for audio takes (B, InputSize, Time)

            # If shape is (B, 1, F, T) -> squeeze -> (B, F, T)
            if x.size(1) == 1:
                x = x.squeeze(1)

        # Now x is likely 3D (B, F, T) or (B, T, F)
        # We need (B, InputSize, Time) where InputSize matches self.tcn.input_size

        if x.dim() == 3:
            # Check which dimension matches input_size
            if x.size(1) == self.input_size:
                # Already (B, InputSize, Time)
                pass
            elif x.size(2) == self.input_size:
                # Is (B, Time, InputSize) -> Transpose to (B, InputSize, Time)
                x = x.transpose(1, 2)
            else:
                # Ambiguous or mismatch, try to infer from common shapes
                # If neither matches, we might have a problem, but let's assume
                # standard (B, F, T) is what we want if F is closer to input_size
                pass

        # TCN forward
        tcn_out = self.tcn(x)

        # Classification
        output = self.fc(tcn_out)

        return cast(torch.Tensor, output)


class TinyConvWakeword(nn.Module):
    """
    Tiny Convolutional Network for wakeword detection with dynamic configuration
    Extremely lightweight architecture suitable for edge devices
    """

    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 1,
        dropout: float = 0.3,
        tcn_num_channels: Optional[list] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize TinyConvWakeword with dynamic channel configuration

        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels
            dropout: Dropout rate
            tcn_num_channels: List of channel sizes for each conv block
            **kwargs: Additional arguments
        """
        super().__init__()

        # Get channel configuration
        if tcn_num_channels is None:
            tcn_num_channels = [16, 32, 64, 64]

        channels = tcn_num_channels
        kernel_size = kwargs.get("kernel_size", 3)
        conv_dropout = kwargs.get("tcn_dropout", dropout)

        # Build dynamic feature extraction layers
        layers: List[nn.Module] = []
        in_channels = input_channels

        for i, out_channels in enumerate(channels):
            # Determine stride (2 for first few layers to downsample, 1 for later layers)
            stride = 2 if i < min(3, len(channels) - 1) else 1

            # Add convolutional block
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,  # Same padding
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

            # Add dropout between conv blocks (except for the last block)
            if i < len(channels) - 1 and conv_dropout > 0:
                layers.append(nn.Dropout2d(conv_dropout))

            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Classifier with final channel count
        final_channels = channels[-1] if channels else 64
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(final_channels, num_classes))

        # NEW: Quantization stubs for QAT support
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Output tensor of shape (batch, num_classes)
        """
        x = self.quant(x)
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


class CDDNNWakeword(nn.Module):
    """
    Context-Dependent Deep Neural Network (CD-DNN)
    Essentially a Multi-Layer Perceptron (MLP) that takes a flattened
    context window of features as input.
    """

    def __init__(
        self,
        input_size: Optional[int] = None,
        hidden_layers: Optional[list] = None,
        num_classes: int = 2,
        dropout: float = 0.3,
        **kwargs: Any,
    ):
        """
        Initialize CD-DNN with dynamic configuration

        Args:
            input_size: Total input size (features * context_frames)
            hidden_layers: List of hidden layer sizes
            num_classes: Number of output classes
            dropout: Dropout rate
            **kwargs: Additional config parameters including:
                - cddnn_hidden_layers: List of hidden layer sizes
                - cddnn_context_frames: Number of context frames
                - cddnn_dropout: Dropout rate for CD-DNN layers
        """
        super().__init__()

        # Use config parameters from kwargs if not provided directly
        if hidden_layers is None:
            hidden_layers = kwargs.get("cddnn_hidden_layers", [512, 256, 128])

        # Override dropout if specified in kwargs
        dropout = kwargs.get("cddnn_dropout", dropout)

        # Calculate input size if not provided
        if input_size is None:
            # Get from kwargs or use default
            feature_size = kwargs.get("input_size", 64)
            context_frames = kwargs.get("cddnn_context_frames", 50)
            input_size = feature_size * context_frames

        layers: list[nn.Module] = []
        in_features = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = hidden_size

        self.network = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch, channels, height, width) or (batch, time, features)
               Will be flattened to (batch, input_size)
        """
        # Flatten input
        x = torch.flatten(x, 1)

        # MLP forward
        x = self.network(x)
        x = self.classifier(x)

        return x

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings (output of last hidden layer)"""
        x = torch.flatten(x, 1)
        x = self.network(x)
        return x


class ConformerWakeword(nn.Module):
    """
    Simplified Conformer-style architecture for wakeword detection.
    Combines Convolutional blocks with Transformer blocks.
    """

    def __init__(
        self,
        input_size: int = 64,
        num_classes: int = 2,
        encoder_dim: int = 144,
        num_layers: int = 4,
        num_heads: int = 4,
        kernel_size: int = 31,
        dropout: float = 0.1,
        **kwargs: Any,
    ):
        """
        Initialize Conformer-style model.

        Args:
            input_size: Input feature dimension
            num_classes: Number of output classes
            encoder_dim: Dimension of the encoder blocks
            num_layers: Number of encoder layers
            num_heads: Number of attention heads
            kernel_size: Kernel size for convolutional blocks
            dropout: Dropout rate
        """
        super().__init__()

        self.input_size = input_size
        self.encoder_dim = encoder_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, encoder_dim), nn.LayerNorm(encoder_dim), nn.Dropout(dropout)
        )

        # Transformer blocks with convolutional modules
        # For simplicity, we use standard Transformer layers but could wrap them
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=num_heads,
            dim_feedforward=encoder_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Convolutional module (applied after transformer or as part of it)
        # Here we add it as a separate block for "Conformer" feel
        self.conv_norm = nn.LayerNorm(encoder_dim)
        self.conv_module = nn.Sequential(
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=encoder_dim),
            nn.BatchNorm1d(encoder_dim),
            nn.GELU(),
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=1),
            nn.Dropout(dropout),
        )

        # Classifier
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(encoder_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Expects (batch, time, features) or (batch, 1, freq, time)
        """
        if x.dim() == 4:
            if x.size(1) == 1:
                x = x.squeeze(1)
            if x.size(1) == self.input_size:
                x = x.transpose(1, 2)

        # Project to encoder dim
        x = self.input_proj(x)  # (B, T, D)

        # Transformer blocks
        x = self.transformer(x)  # (B, T, D)

        # Conv module (needs B, D, T for convolutions, but LayerNorm needs B, T, D)
        residual = x
        x = self.conv_norm(x)
        x = x.transpose(1, 2)
        x = self.conv_module(x)
        x = x.transpose(1, 2) + residual  # Residual in (B, T, D)

        # Classify (pool needs B, D, T)
        x = x.transpose(1, 2)
        logits = self.classifier(x)

        return logits

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings"""
        if x.dim() == 4:
            if x.size(1) == 1:
                x = x.squeeze(1)
            if x.size(1) == self.input_size:
                x = x.transpose(1, 2)

        x = self.input_proj(x)
        x = self.transformer(x)

        residual = x
        x = self.conv_norm(x)
        x = x.transpose(1, 2)
        x = self.conv_module(x)
        x = x.transpose(1, 2) + residual

        x = x.transpose(1, 2)
        # Pool to get fixed size embedding
        embedding = F.adaptive_avg_pool1d(x, 1).flatten(1)
        return embedding


def create_model(architecture: str, num_classes: int = 2, pretrained: bool = False, **kwargs: Any) -> nn.Module:
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
        return ResNet18Wakeword(num_classes=num_classes, pretrained=pretrained, **kwargs)  # Pass all kwargs to model

    elif architecture == "mobilenetv3":
        return MobileNetV3Wakeword(num_classes=num_classes, pretrained=pretrained, **kwargs)  # Pass all kwargs to model

    elif architecture == "lstm":
        return LSTMWakeword(num_classes=num_classes, **kwargs)  # Pass all kwargs to model

    elif architecture == "gru":
        return GRUWakeword(num_classes=num_classes, **kwargs)  # Pass all kwargs to model

    elif architecture == "tcn":
        return TCNWakeword(num_classes=num_classes, **kwargs)  # Pass all kwargs to model

    elif architecture == "tiny_conv":
        return TinyConvWakeword(num_classes=num_classes, **kwargs)  # Pass all kwargs to model

    elif architecture == "cd_dnn":
        return CDDNNWakeword(num_classes=num_classes, **kwargs)  # Pass all kwargs to model

    elif architecture == "conformer":
        return ConformerWakeword(num_classes=num_classes, **kwargs)

    elif architecture == "wav2vec2":
        from src.models.huggingface import Wav2VecWakeword

        return Wav2VecWakeword(num_classes=num_classes, **kwargs)

    else:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Supported: resnet18, mobilenetv3, lstm, gru, tcn, tiny_conv, cd_dnn, conformer, wav2vec2"
        )


if __name__ == "__main__":
    # Test model creation
    print("Model Architectures Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Test each architecture
    architectures = ["resnet18", "mobilenetv3", "lstm", "gru", "tiny_conv", "cd_dnn"]

    for arch in architectures:
        print(f"\nTesting {arch}...")

        model = create_model(arch, num_classes=2, pretrained=False)
        model = model.to(device)

        # Test forward pass
        if arch in ["resnet18", "mobilenetv3", "tiny_conv"]:
            # 2D input (batch, channels, height, width)
            test_input = torch.randn(2, 1, 64, 50).to(device)
        else:
            # Sequential input (batch, time, features)
            test_input = torch.randn(2, 50, 64).to(device)

        output = model(test_input)
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  ✅ {arch} works correctly")

    print("\n✅ All architectures tested successfully")
    print("Model architectures module loaded successfully")
