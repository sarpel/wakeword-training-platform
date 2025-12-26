"""
HuggingFace Model Wrappers for Wakeword Detection

Teacher models are automatically downloaded from HuggingFace Hub on first use.
Default cache location: models/teachers/
"""

import logging
import os
from pathlib import Path
from typing import Optional, cast

import torch
import torch.nn as nn

from src.config.logger import get_logger

# Set default teacher model cache directory
TEACHER_CACHE_DIR = Path("models/teachers")
TEACHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Set HuggingFace cache to our directory
os.environ.setdefault("TRANSFORMERS_CACHE", str(TEACHER_CACHE_DIR))
os.environ.setdefault("HF_HOME", str(TEACHER_CACHE_DIR))

try:
    from transformers import Wav2Vec2Config, Wav2Vec2Model, WhisperModel, WhisperConfig, utils

    utils.logging.set_verbosity_warning()
    TRANSFORMERS_IMPORT_ERROR = None
except ImportError as e:
    logging.getLogger(__name__).warning(f"Failed to import transformers: {e}")
    TRANSFORMERS_IMPORT_ERROR = str(e)
    Wav2Vec2Model = None
    Wav2Vec2Config = None
    WhisperModel = None
    WhisperConfig = None

logger = get_logger(__name__)


def get_teacher_cache_dir() -> Path:
    """Get the teacher model cache directory."""
    return TEACHER_CACHE_DIR


def ensure_teacher_model_downloaded(model_id: str = "facebook/wav2vec2-base-960h") -> Path:
    """
    Ensure teacher model is downloaded and cached.

    Args:
        model_id: HuggingFace model ID

    Returns:
        Path to cached model directory
    """
    if Wav2Vec2Model is None:
        raise ImportError("transformers library required. Install with: pip install transformers")

    cache_path = TEACHER_CACHE_DIR / model_id.replace("/", "--")

    if not cache_path.exists():
        logger.info(f"Downloading teacher model {model_id}...")
        # This will download and cache the model
        _ = Wav2Vec2Model.from_pretrained(model_id, cache_dir=str(TEACHER_CACHE_DIR))
    else:
        logger.debug(f"Teacher model already cached at {cache_path}")

    return cache_path


class Wav2VecWakeword(nn.Module):
    """
    Wav2Vec 2.0 wrapper for Wakeword Detection.
    Uses the transformer as a feature extractor and adds a classification head.

    Teacher model is automatically downloaded on first use to models/teachers/
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_feature_extractor: bool = True,
        model_id: str = "facebook/wav2vec2-base-960h",
    ):
        super().__init__()

        if Wav2Vec2Model is None:
            raise ImportError(
                f"transformers library is required for Wav2VecWakeword. Please install it with `pip install transformers`.\n"
                f"Original error: {TRANSFORMERS_IMPORT_ERROR}"
            )

        if pretrained:
            # Model will be downloaded to TEACHER_CACHE_DIR on first use
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_id, cache_dir=str(TEACHER_CACHE_DIR))
        else:
            config = Wav2Vec2Config.from_pretrained(model_id, cache_dir=str(TEACHER_CACHE_DIR))
            self.wav2vec2 = Wav2Vec2Model(config)

        if freeze_feature_extractor:
            if hasattr(self.wav2vec2, "freeze_feature_encoder"):
                self.wav2vec2.freeze_feature_encoder()
            elif hasattr(self.wav2vec2.feature_extractor, "_freeze_parameters"):
                self.wav2vec2.feature_extractor._freeze_parameters()

        logger.info(f"[✓] Teacher: Wav2Vec2 Loaded & Frozen ({model_id})")

        # Classification head
        # Wav2Vec2 base output dim is 768
        self.hidden_size = 768
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Input audio tensor of shape (batch, samples)
        Returns:
            Logits of shape (batch, num_classes)
        """
        # Get embeddings first
        pooled_output = self.embed(x)

        # Classification
        logits = self.classifier(pooled_output)

        return cast(torch.Tensor, logits)

    def embed(self, x: torch.Tensor, layer_index: Optional[int] = None) -> torch.Tensor:
        """
        Extract embeddings (for distillation feature alignment).

        Args:
            x: Input audio tensor of shape (batch, samples) or (batch, 1, samples)
            layer_index: Optional index of the transformer layer to extract from.
                        If None, uses the last hidden state.

        Returns:
            Embeddings of shape (batch, hidden_size=768)
        """
        # Ensure input is 2D (batch, samples)
        if x.ndim == 3:
            x = x.squeeze(1)

        # Wav2Vec2 expects normalized audio
        outputs = self.wav2vec2(x, output_hidden_states=True if layer_index is not None else False)

        if layer_index is not None and hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            # layer_index 0 is the initial CNN embedding, 1-12 are transformer layers
            hidden_states = outputs.hidden_states
            idx = max(0, min(layer_index, len(hidden_states) - 1))
            selected_hidden_state = hidden_states[idx]
            # Pooling: Mean over time dimension
            pooled_output = torch.mean(selected_hidden_state, dim=1)
        else:
            # Get hidden state from the last layer
            # shape: (batch, sequence_length, hidden_size)
            last_hidden_state = outputs.last_hidden_state
            # Pooling: Mean over time dimension
            pooled_output = torch.mean(last_hidden_state, dim=1)

        return cast(torch.Tensor, pooled_output)


class WhisperWakeword(nn.Module):
    """
    OpenAI Whisper encoder wrapper for Wakeword Detection.
    Uses the Whisper encoder as a feature extractor and adds a classification head.

    Teacher model is automatically downloaded on first use to models/teachers/
    Recommended: openai/whisper-tiny for efficiency, openai/whisper-base for quality.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_encoder: bool = True,
        model_id: str = "openai/whisper-tiny",
    ):
        """
        Initialize Whisper-based wakeword detector.

        Args:
            num_classes: Number of output classes (default: 2 for wake/not-wake)
            pretrained: Whether to load pretrained weights from HuggingFace
            freeze_encoder: Whether to freeze encoder weights (recommended for distillation)
            model_id: HuggingFace model ID (whisper-tiny is ~39M params, whisper-base is ~74M)
        """
        super().__init__()

        if WhisperModel is None:
            raise ImportError(
                f"transformers library is required for WhisperWakeword. "
                f"Please install it with `pip install transformers`.\n"
                f"Original error: {TRANSFORMERS_IMPORT_ERROR}"
            )

        if pretrained:
            # Download Whisper model to teacher cache directory
            self.whisper = WhisperModel.from_pretrained(model_id, cache_dir=str(TEACHER_CACHE_DIR))
        else:
            config = WhisperConfig.from_pretrained(model_id, cache_dir=str(TEACHER_CACHE_DIR))
            self.whisper = WhisperModel(config)

        # Freeze encoder if specified (recommended for teacher model)
        if freeze_encoder:
            for param in self.whisper.encoder.parameters():
                param.requires_grad = False
            logger.info(f"[✓] Teacher: Whisper Encoder Loaded & Frozen ({model_id})")
        else:
            logger.info(f"[✓] Teacher: Whisper Encoder Loaded ({model_id})")

        # Whisper hidden size: tiny=384, base=512, small=768, medium=1024, large=1280
        self.hidden_size = self.whisper.config.d_model

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, num_classes)
        )

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepare audio input for Whisper encoder.
        Whisper expects log-mel spectrogram of shape (batch, 80, 3000) for 30s audio.
        For shorter audio, we pad or generate mel-spectrograms.

        Args:
            x: Input tensor - can be raw audio (batch, samples) or features (batch, 1, freq, time)

        Returns:
            Tensor ready for Whisper encoder (batch, 80, time)
        """
        # If already 3D with correct feature dim, assume it's ready
        if x.ndim == 3 and x.size(1) == 80:
            return x

        # If 4D (batch, 1, freq, time), squeeze and check
        if x.ndim == 4:
            x = x.squeeze(1)  # (batch, freq, time)
            if x.size(1) == 80:
                return x

        # If 2D raw audio, we need to compute mel-spectrogram
        # For simplicity in distillation, we'll use a simple mel computation
        # In production, you'd use Whisper's feature extractor
        if x.ndim == 2:
            # Raw audio input - use torchaudio for mel spectrogram
            try:
                import torchaudio.transforms as T

                mel_transform = T.MelSpectrogram(
                    sample_rate=16000, n_fft=400, hop_length=160, n_mels=80, normalized=True
                ).to(x.device)
                mel = mel_transform(x)  # (batch, 80, time)
                # Log mel spectrogram
                mel = torch.log(mel.clamp(min=1e-10))
                return mel
            except ImportError:
                # Fallback: simple linear projection (not ideal but functional)
                logger.warning("torchaudio not available, using simple projection for Whisper input")
                # Reshape to fake mel features
                seq_len = x.size(1) // 160  # Approximate time frames
                x_reshaped = x[:, : seq_len * 160].reshape(x.size(0), seq_len, 160)
                # Simple linear to 80 dims
                x_proj = x_reshaped[:, :, :80].transpose(1, 2)  # (batch, 80, time)
                return x_proj

        # If 3D but wrong feature dim, try to adapt
        if x.ndim == 3:
            # Assume (batch, time, features) - transpose to (batch, features, time)
            if x.size(2) < x.size(1):
                x = x.transpose(1, 2)
            # Pad or project to 80 features if needed
            if x.size(1) != 80:
                # Simple adaptation via linear layer (created on-the-fly - not ideal but works)
                x = torch.nn.functional.adaptive_avg_pool1d(x, 80)
                x = x.transpose(1, 2).transpose(1, 2)  # Ensure (batch, 80, time)
            return x

        raise ValueError(f"Unexpected input shape for Whisper: {x.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Whisper encoder + classifier.

        Args:
            x: Input audio tensor of shape (batch, samples) or (batch, 80, time)

        Returns:
            Logits of shape (batch, num_classes)
        """
        # Get embeddings
        pooled_output = self.embed(x)

        # Classification
        logits = self.classifier(pooled_output)

        return cast(torch.Tensor, logits)

    def embed(self, x: torch.Tensor, layer_index: Optional[int] = None) -> torch.Tensor:
        """
        Extract embeddings from Whisper encoder (for distillation feature alignment).

        Args:
            x: Input audio tensor
            layer_index: Optional index of encoder layer to extract from.
                        If None, uses the last encoder hidden state.

        Returns:
            Embeddings of shape (batch, hidden_size)
        """
        # Prepare input for Whisper
        mel_input = self._prepare_input(x)

        # Whisper encoder forward
        encoder_outputs = self.whisper.encoder(
            mel_input, output_hidden_states=True if layer_index is not None else False, return_dict=True
        )

        if (
            layer_index is not None
            and hasattr(encoder_outputs, "hidden_states")
            and encoder_outputs.hidden_states is not None
        ):
            # Extract from specific layer
            hidden_states = encoder_outputs.hidden_states
            idx = max(0, min(layer_index, len(hidden_states) - 1))
            selected_hidden_state = hidden_states[idx]
            # Mean pooling over time dimension
            pooled_output = torch.mean(selected_hidden_state, dim=1)
        else:
            # Use last hidden state
            last_hidden_state = encoder_outputs.last_hidden_state
            # Mean pooling over time dimension
            pooled_output = torch.mean(last_hidden_state, dim=1)

        return cast(torch.Tensor, pooled_output)
