"""
HuggingFace Model Wrappers for Wakeword Detection

Teacher models are automatically downloaded from HuggingFace Hub on first use.
Default cache location: models/teachers/
"""

import logging
import os
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn

# Set default teacher model cache directory
TEACHER_CACHE_DIR = Path("models/teachers")
TEACHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Set HuggingFace cache to our directory
os.environ.setdefault("TRANSFORMERS_CACHE", str(TEACHER_CACHE_DIR))
os.environ.setdefault("HF_HOME", str(TEACHER_CACHE_DIR))

try:
    from transformers import Wav2Vec2Config, Wav2Vec2Model, utils

    utils.logging.set_verbosity_error()  # Silence initialization warnings
    TRANSFORMERS_IMPORT_ERROR = None
except ImportError as e:
    logging.getLogger(__name__).warning(f"Failed to import transformers: {e}")
    TRANSFORMERS_IMPORT_ERROR = str(e)
    Wav2Vec2Model = None
    Wav2Vec2Config = None

logger = logging.getLogger(__name__)


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

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings (for distillation feature alignment).

        Args:
            x: Input audio tensor of shape (batch, samples) or (batch, 1, samples)

        Returns:
            Embeddings of shape (batch, hidden_size=768)
        """
        # Ensure input is 2D (batch, samples)
        if x.ndim == 3:
            x = x.squeeze(1)

        # Wav2Vec2 expects normalized audio
        outputs = self.wav2vec2(x)

        # Get hidden state from the last layer
        # shape: (batch, sequence_length, hidden_size)
        last_hidden_state = outputs.last_hidden_state

        # Pooling: Mean over time dimension
        # shape: (batch, hidden_size)
        pooled_output = torch.mean(last_hidden_state, dim=1)

        return cast(torch.Tensor, pooled_output)
