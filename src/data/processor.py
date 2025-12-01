"""
GPU Audio Processor
Handles feature extraction and normalization on GPU for training pipeline.
"""
import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Any, Optional, cast

from src.data.feature_extraction import FeatureExtractor
from src.data.cmvn import CMVN

logger = logging.getLogger(__name__)

from src.config.defaults import WakewordConfig

class AudioProcessor(nn.Module):
    """
    GPU-accelerated Audio Processor.
    Handles feature extraction (Mel/MFCC) and CMVN on GPU.
    """
    def __init__(self, config: WakewordConfig, cmvn_path: Optional[Path] = None, device: str = "cuda") -> None:
        super().__init__()
        self.config = config
        self.device = device
        
        # Feature Extractor
        self.feature_extractor = FeatureExtractor(
            sample_rate=config.data.sample_rate,
            feature_type=config.data.feature_type,  # type: ignore
            n_mels=config.data.n_mels,
            n_mfcc=config.data.n_mfcc,
            n_fft=config.data.n_fft,
            hop_length=config.data.hop_length,
            device=device
        )
        
        # CMVN
        self.cmvn = None
        # Check if we should use CMVN (assuming normalize_audio implies normalization)
        # Or check specific flag if available. 
        # Existing code passed cmvn_path, so we assume usage if path exists.
        if cmvn_path:
             self.cmvn = CMVN(stats_path=cmvn_path)
             self.cmvn.to(device)
             logger.info(f"CMVN initialized on {device}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Process inputs (raw audio or features).
        
        Args:
            inputs: Raw audio (B, T) or (B, 1, T) OR Features (B, C, F, T)
            
        Returns:
            Processed features (B, C, F, T)
        """
        # If inputs are raw audio (B, T) or (B, 1, T)
        # Check if input is raw audio or features
        # Raw audio: (Batch, Samples) or (Batch, 1, Samples) -> ndim 2 or 3
        # Features: (Batch, Channel, Freq, Time) -> ndim 4
        
        if inputs.ndim <= 3:
            # Raw audio processing pipeline
            
            # 1. Resample if needed (handled by torchaudio.transforms.Resample usually, 
            # but here we assume inputs are already at target_sr or we rely on augmentation)
            # Actually, AudioProcessor usually expects raw audio and does feature extraction.
            
            # Ensure (Batch, 1, Samples)
            if inputs.ndim == 2:
                inputs = inputs.unsqueeze(1)
            # FeatureExtractor expects (B, T) or (B, 1, T)
            features = self.feature_extractor(inputs)
        else:
            # Already features (B, C, F, T)
            features = inputs
            
        # Apply CMVN
        if self.cmvn is not None:
            # CMVN expects (B, C, F, T)
            features = self.cmvn.normalize(features)
            
        return cast(torch.Tensor, features)
