"""
HuggingFace Model Wrappers for Wakeword Detection
"""
import logging
import torch
import torch.nn as nn
from typing import Optional, cast

try:
    from transformers import Wav2Vec2Model, Wav2Vec2Config
except ImportError:
    Wav2Vec2Model = None
    Wav2Vec2Config = None

logger = logging.getLogger(__name__)

class Wav2VecWakeword(nn.Module):
    """
    Wav2Vec 2.0 wrapper for Wakeword Detection.
    Uses the transformer as a feature extractor and adds a classification head.
    """
    
    def __init__(
        self, 
        num_classes: int = 2, 
        pretrained: bool = True, 
        freeze_feature_extractor: bool = True,
        model_id: str = "facebook/wav2vec2-base-960h"
    ):
        super().__init__()
        
        if Wav2Vec2Model is None:
            raise ImportError("transformers library is required for Wav2VecWakeword. Please install it with `pip install transformers`.")

        logger.info(f"Initializing Wav2VecWakeword with {model_id}")
        
        if pretrained:
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_id)
        else:
            config = Wav2Vec2Config.from_pretrained(model_id)
            self.wav2vec2 = Wav2Vec2Model(config)
            
        if freeze_feature_extractor:
            self.wav2vec2.feature_extractor._freeze_parameters()
            logger.info("Wav2Vec2 feature extractor frozen")

        # Classification head
        # Wav2Vec2 base output dim is 768
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Input audio tensor of shape (batch, samples)
        Returns:
            Logits of shape (batch, num_classes)
        """
        # Ensure input is 2D (batch, samples)
        if x.ndim == 3:
            x = x.squeeze(1)
            
        # Wav2Vec2 expects normalized audio
        # We assume input is already normalized by DataLoader
        
        outputs = self.wav2vec2(x)
        
        # Get hidden state from the last layer
        # shape: (batch, sequence_length, hidden_size)
        last_hidden_state = outputs.last_hidden_state
        
        # Pooling: Mean over time dimension
        # shape: (batch, hidden_size)
        pooled_output = torch.mean(last_hidden_state, dim=1)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return cast(torch.Tensor, logits)
