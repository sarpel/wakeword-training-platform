"""
HuggingFace Model Wrappers for Wakeword Detection

Teacher models are automatically downloaded from HuggingFace Hub on first use.
Default cache location: models/teachers/
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional, cast

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
    from transformers import Wav2Vec2Config, Wav2Vec2Model, WhisperConfig, WhisperModel, utils

    utils.logging.set_verbosity_warning()
    TRANSFORMERS_IMPORT_ERROR = None
except ImportError as e:
    logging.getLogger(__name__).warning(f"Failed to import transformers: {e}")
    TRANSFORMERS_IMPORT_ERROR = str(e)
    Wav2Vec2Model: Any = None  # type: ignore[no-redef]
    Wav2Vec2Config: Any = None  # type: ignore[no-redef]
    WhisperModel: Any = None  # type: ignore[no-redef]
    WhisperConfig: Any = None  # type: ignore[no-redef]

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

        # Cache MelSpectrogram transform (lazy initialization on first use)
        self._mel_transform = None

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepare audio input for Whisper encoder.
        Whisper expects log-mel spectrogram of shape (batch, 80, 3000) for 30s audio.
        Uses Whisper's exact preprocessing: log10, max clamp, normalization to ~[-1, 1].

        Args:
            x: Input tensor - can be:
              - Raw audio: (batch, samples) or (batch, 1, samples)
              - Mel features: (batch, 80, time) or (batch, 1, 80, time)

        Returns:
            Tensor ready for Whisper encoder (batch, 80, 3000)
        """
        mel = None

        # Helper: Convert raw audio to Whisper-compatible log-mel spectrogram
        def compute_mel(audio: torch.Tensor) -> torch.Tensor:
            """
            Convert raw audio [batch, samples] to Whisper-compatible log-mel [batch, 80, time].
            Matches OpenAI's Whisper preprocessing exactly.
            """
            import torchaudio.transforms as T

            # Check for NaNs in raw audio
            if torch.isnan(audio).any() or torch.isinf(audio).any():
                logger.warning("NaN/Inf detected in raw audio input to Whisper, replacing with silence")
                audio = torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

            # Whisper mel spectrogram parameters:
            # - 80 mel bins, 400 sample window (25ms), 160 sample hop (10ms)
            # - NOT normalized in torchaudio (normalized=False) - we normalize manually
            # Cache the transform to avoid repeated instantiation
            if self._mel_transform is None:
                self._mel_transform = T.MelSpectrogram(
                    sample_rate=16000, n_fft=400, hop_length=160, n_mels=80, normalized=False
                )
            mel_transform = self._mel_transform.to(audio.device)
            mel_out = mel_transform(audio)  # (batch, 80, time)

            # === WHISPER PREPROCESSING (matching OpenAI's implementation) ===
            # Step 1: Log10 scale (NOT natural log!)
            log_spec = torch.log10(torch.clamp(mel_out, min=1e-10))

            # Step 2: Max clamp - prevent extreme negative values
            # Clamp to max - 8.0 (where max is per-sample for batch processing)
            max_val = log_spec.amax(dim=(-2, -1), keepdim=True)
            log_spec = torch.maximum(log_spec, max_val - 8.0)

            # Step 3: Normalize to ~[-1, 1] range
            # This is the standard Whisper normalization
            log_spec = (log_spec + 4.0) / 4.0

            # Final safety clamp to valid range (usually -1.5 to 1.5)
            # This prevents any numerical slips from blowing up later layers
            return torch.clamp(log_spec, min=-5.0, max=5.0)

        # Case 1: Already correct mel features (batch, 80, time)
        # Note: If pre-computed mel is already in Whisper format, use directly
        if x.ndim == 3 and x.size(1) == 80:
            mel = x

        # Case 2: Mel features with channel dim (batch, 1, 80, time)
        elif x.ndim == 4 and x.size(1) == 1 and x.size(2) == 80:
            mel = x.squeeze(1)  # (batch, 80, time)

        # Case 3: Raw audio with channel dim (batch, 1, samples) - COMMON CASE
        # Detected when: 3D, channel=1, and time dimension is > 1000 (audio samples, not mel frames)
        elif x.ndim == 3 and x.size(1) == 1 and x.size(2) > 1000:
            # Squeeze channel and compute mel
            audio = x.squeeze(1)  # (batch, samples)
            mel = compute_mel(audio)

        # Case 4: Raw audio without channel dim (batch, samples)
        elif x.ndim == 2:
            mel = compute_mel(x)

        # Case 5: Other 4D shapes (batch, 1, freq, time) with freq != 80
        elif x.ndim == 4:
            x = x.squeeze(1)  # (batch, freq, time)
            if x.size(1) == 80:
                mel = x
            else:
                # Unexpected feature dimension - try to adapt
                logger.warning(f"Whisper received unexpected 4D input shape {x.shape}, attempting adaptation")
                mel = torch.nn.functional.interpolate(x.unsqueeze(1), size=(80, x.size(2)), mode="bilinear").squeeze(1)

        # Case 6: Other 3D shapes - likely (batch, time, features), transpose to (batch, features, time)
        elif x.ndim == 3:
            if x.size(1) > x.size(2):
                # Looks like (batch, time, features), transpose
                x = x.transpose(1, 2)
            # Check if it's now 80 features
            if x.size(1) == 80:
                mel = x
            else:
                # Unexpected - try interpolation
                logger.warning(f"Whisper received unexpected 3D input shape after transpose {x.shape}")
                mel = torch.nn.functional.interpolate(x.unsqueeze(1), size=(80, x.size(2)), mode="bilinear").squeeze(1)

        if mel is None:
            raise ValueError(f"Unexpected input shape for Whisper: {x.shape}")

        # CRITICAL: Whisper encoder requires exactly 3000 time frames
        # (corresponds to 30 seconds of audio at 100 frames/sec)
        # Pad or truncate to exactly 3000 frames
        target_len = 3000
        current_len = mel.size(2)

        if current_len < target_len:
            # Pad with normalized silence value (0.0 in normalized scale = near-silence)
            padding = target_len - current_len
            mel = torch.nn.functional.pad(mel, (0, padding), mode="constant", value=0.0)
        elif current_len > target_len:
            # Truncate to target length
            mel = mel[:, :, :target_len]

        return mel

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
