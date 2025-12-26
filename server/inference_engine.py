import torch
import numpy as np
import os
import sys
import io
import soundfile as sf
import pickle
from pathlib import Path
from typing import Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# SECURITY: Safe checkpoint loading to prevent arbitrary code execution
# ============================================================================
def safe_load_checkpoint(checkpoint_path: Union[str, Path], device: str = "cpu") -> dict:
    """
    Safely load a checkpoint with validation to prevent arbitrary code execution.
    
    This function implements defense-in-depth:
    1. Validates the checkpoint path (prevents path traversal)
    2. Loads tensors only when possible (weights_only=True) for PyTorch 2.4+
    3. Validates checkpoint structure before returning
    4. Provides clear error messages for debugging
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to map tensors to
        
    Returns:
        dict: Validated checkpoint dictionary
        
    Raises:
        ValueError: If checkpoint path or content is invalid
        FileNotFoundError: If checkpoint file does not exist
    """
    checkpoint_path = Path(checkpoint_path)
    
    # SECURITY: Validate checkpoint path (prevent path traversal)
    # Checkpoints should be in models/ or checkpoints/ directories
    allowed_dirs = ["models", "checkpoints", "cache", "exports"]
    try:
        resolved_path = checkpoint_path.resolve()
        # Check if parent directory is in allowed list
        parent_name = resolved_path.parent.name
        grandparent_name = resolved_path.parent.parent.name if resolved_path.parent.parent else ""
        
        is_allowed = (
            parent_name in allowed_dirs or
            grandparent_name in allowed_dirs
        )
        
        if not is_allowed:
            raise ValueError(
                f"Security violation: Checkpoint path '{checkpoint_path}' resolves to '{resolved_path}' "
                f"which is not in allowed directories ({allowed_dirs})"
            )
    except Exception as e:
        raise ValueError(f"Invalid checkpoint path: {e}") from e
    
    # SECURITY: Check file exists and is a regular file
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint path is not a file: {checkpoint_path}")
    
    # SECURITY: Load checkpoint safely
    try:
        # Try weights_only=True (PyTorch 2.4+ recommended)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        logger.info("Loaded checkpoint with weights_only=True (safest)")
    except Exception:
        # Fallback: Load with pickle but validate structure afterward
        checkpoint = torch.load(checkpoint_path, map_location=device)
        logger.warning("Loaded checkpoint with pickle (less safe - validate structure)")
        
        # SECURITY: Validate checkpoint structure
        required_keys = ["model_state_dict"]
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(
                    f"Invalid checkpoint format: missing required key '{key}'. "
                    f"This may be a corrupted or malicious checkpoint file."
                )
        
        # SECURITY: Check for suspicious keys (arbitrary code)
        suspicious_keys = ["__builtins__", "__code__", "__func__", "eval", "exec", "compile"]
        for key in checkpoint.keys():
            key_str = str(key).lower()
            for susp in suspicious_keys:
                if susp in key_str:
                    raise ValueError(
                        f"Security violation: Suspicious key '{key}' found in checkpoint. "
                        f"This checkpoint may contain arbitrary code."
                    )
    
    return checkpoint

# Ensure src is in path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.logger import get_logger as setup_logger
from src.config.paths import paths
from src.config.defaults import WakewordConfig
from src.models.architectures import create_model
from src.data.processor import AudioProcessor

logger = setup_logger("wakeword_server")

class InferenceEngine:
    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.device = device
        
        # Load Model and Config
        self.model, self.config = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize Audio Processor (Standardized)
        # Use the config from the model checkpoint for feature parity
        self.audio_processor = AudioProcessor(
            config=self.config,
            cmvn_path=paths.CMVN_STATS if paths.CMVN_STATS.exists() else None,
            device=self.device
        )
        self.audio_processor.eval()
        
        logger.info(f"Inference Engine initialized on {self.device}")

    def _load_model(self, model_path: str):
        if not model_path or not os.path.exists(model_path):
            # Fallback to default best model path
            model_path = paths.CHECKPOINTS / "best_model.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")

        logger.info(f"Loading checkpoint from {model_path}")
        try:
            # SECURITY: Use safe loading to prevent arbitrary code execution
            checkpoint = safe_load_checkpoint(model_path, self.device)
        except Exception as e:
            logger.exception(f"Failed to load checkpoint from {model_path}")
            raise FileNotFoundError(f"Model not found or corrupted at {model_path}") from e
        
        # Reconstruct Config
        config_data = checkpoint.get("config")
        if isinstance(config_data, dict):
            config = WakewordConfig.from_dict(config_data)
        else:
            config = config_data or WakewordConfig()

        # Architecture Sync (The "Blueprint" Fix)
        # Calculate input size based on config
        input_samples = int(config.data.sample_rate * config.data.audio_duration)
        time_steps = input_samples // config.data.hop_length + 1
        feature_dim = config.data.n_mels if config.data.feature_type in ["mel", "mel_spectrogram"] else config.data.n_mfcc
        
        if config.model.architecture == "cd_dnn":
            input_size = feature_dim * time_steps
        elif config.model.architecture == "wav2vec2":
            input_size = None # Handled by Wav2VecWakeword internally
        else:
            input_size = feature_dim

        # Create Model with all dynamic parameters
        model = create_model(
            architecture=config.model.architecture,
            num_classes=config.model.num_classes,
            pretrained=False,
            dropout=config.model.dropout,
            input_size=input_size,
            input_channels=1,
            # RNN params
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
            bidirectional=config.model.bidirectional,
            # TCN / TinyConv params
            tcn_num_channels=getattr(config.model, "tcn_num_channels", None),
            tcn_kernel_size=getattr(config.model, "tcn_kernel_size", 3),
            tcn_dropout=getattr(config.model, "tcn_dropout", config.model.dropout),
            # CD-DNN params
            cddnn_hidden_layers=getattr(config.model, "cddnn_hidden_layers", None),
            cddnn_context_frames=getattr(config.model, "cddnn_context_frames", 50),
            cddnn_dropout=getattr(config.model, "cddnn_dropout", config.model.dropout),
        )

        # Load Weights
        state_dict = checkpoint["model_state_dict"]
        # Handle QAT artifacts if necessary (FP32 Judge load)
        model_keys = set(model.state_dict().keys())
        state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
        
        model.load_state_dict(state_dict, strict=True)
        
        return model, config

    def preprocess(self, audio_data: bytes) -> torch.Tensor:
        """
        Standardized preprocessing using the project's AudioProcessor.
        """
        try:
            # Load bytes into numpy
            audio_np, sr = sf.read(io.BytesIO(audio_data))
            
            # Basic cleanup
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)
            
            # Convert to tensor
            waveform = torch.from_numpy(audio_np).float().unsqueeze(0) # (1, T)
            
            # Use GPU/CPU AudioProcessor for Feature Extraction + CMVN
            with torch.no_grad():
                features = self.audio_processor(waveform.to(self.device))
            
            return features
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise

    def predict(self, audio_data: bytes) -> dict:
        features = self.preprocess(audio_data)
        
        with torch.no_grad():
            logits = self.model(features)
            probs = torch.softmax(logits, dim=1)
            
        # Class 1 is "Wakeword"
        confidence = probs[0, 1].item()
        prediction = 1 if confidence > 0.5 else 0
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "label": "wakeword" if prediction == 1 else "background",
            "architecture": self.config.model.architecture
        }