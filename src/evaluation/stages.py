"""
Modular Inference Stages for Distributed Cascade Architecture.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from src.evaluation.types import StageBase
from src.data.feature_extraction import FeatureExtractor

class SentryInferenceStage(StageBase):
    """
    Sentry Stage: Ultra-low power edge detection (MobileNetV3).
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        name: str = "sentry",
        threshold: float = 0.5,
        device: str = "cpu",
        feature_extractor: Optional[FeatureExtractor] = None
    ):
        self.model = model.to(device)
        self.model.eval()
        self._name = name
        self.threshold = threshold
        self.device = device
        
        # Initialize default feature extractor if not provided
        if feature_extractor is None:
            self.feature_extractor = FeatureExtractor(device=device)
        else:
            self.feature_extractor = feature_extractor.to(device)
            
    @property
    def name(self) -> str:
        return self._name
        
    def predict(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Run Sentry prediction.
        
        Args:
            audio: 1D numpy array of audio samples.
            
        Returns:
            Dictionary with confidence and detection status.
        """
        # Convert to tensor
        waveform = torch.from_numpy(audio).float().to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(waveform)
            
            # Ensure 4D for CNN: (batch, 1, freq, time)
            if features.dim() == 3:
                features = features.unsqueeze(0)
            
            # Model inference
            logits = self.model(features)
            probs = torch.softmax(logits, dim=1)
            confidence = probs[0, 1].item() # Assuming class 1 is positive
            
        return {
            "confidence": confidence,
            "detected": confidence >= self.threshold
        }
