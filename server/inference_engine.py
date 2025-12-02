import torch
import numpy as np
import os
import sys
import io
import soundfile as sf
from pathlib import Path

# Ensure src is in path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.logger import get_logger as setup_logger

try:
    from src.models.huggingface import Wav2VecWakeword
except ImportError:
    # Fallback if src is not available (e.g., simplified docker build)
    # In a real scenario, we'd ensure src is installed or copied
    pass

logger = setup_logger("wakeword_server")

class InferenceEngine:
    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.device = device
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Inference Engine initialized on {self.device}")

    def _load_model(self, model_path: str):
        # Initialize model architecture
        # We assume the Judge is always Wav2Vec2
        try:
            model = Wav2VecWakeword(num_classes=2, pretrained=False)
        except NameError:
            raise ImportError("Could not import Wav2VecWakeword. Ensure src is in python path.")

        if model_path and os.path.exists(model_path):
            logger.info(f"Loading weights from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            # Filter out QAT keys if loading into a non-QAT model
            model_keys = set(model.state_dict().keys())
            unexpected_keys = set(state_dict.keys()) - model_keys
            if unexpected_keys:
                logger.warning(f"Filtering {len(unexpected_keys)} unexpected keys (likely QAT artifacts)")
                for key in unexpected_keys:
                    del state_dict[key]

            model.load_state_dict(state_dict)
        else:
            logger.error(f"Model path {model_path} not found.")
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        return model

    def preprocess(self, audio_data: bytes) -> torch.Tensor:
        """
        Convert audio bytes (WAV or PCM) to float tensor.
        """
        try:
            # Try reading as WAV/FLAC/etc with soundfile
            audio_np, sample_rate = sf.read(io.BytesIO(audio_data))
            
            # If stereo, convert to mono
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)
                
            # Resample if needed (assuming 16kHz required)
            if sample_rate != 16000:
                import librosa
                audio_np = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=16000)
                
        except Exception:
            # Fallback to raw PCM16 if soundfile fails (e.g., no header)
            # Assumes 16kHz mono PCM16
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            audio_np = audio_np.astype(np.float32) / 32768.0

        # Convert to tensor
        tensor = torch.from_numpy(audio_np).float().unsqueeze(0) # (1, T)
        return tensor.to(self.device)

    def predict(self, audio_data: bytes) -> dict:
        tensor = self.preprocess(audio_data)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            
        # Class 1 is "Wakeword"
        confidence = probs[0, 1].item()
        prediction = 1 if confidence > 0.5 else 0
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "label": "wakeword" if prediction == 1 else "background"
        }
