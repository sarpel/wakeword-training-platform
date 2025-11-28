import torch
import numpy as np
import logging
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.models.huggingface import Wav2VecWakeword
except ImportError:
    # Fallback if src is not available (e.g., simplified docker build)
    # In a real scenario, we'd ensure src is installed or copied
    pass

logger = logging.getLogger("wakeword_server")

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
            checkpoint = torch.load(model_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            logger.warning(f"Model path {model_path} not found. Using random weights (for testing).")
            
        return model

    def preprocess(self, audio_data: bytes) -> torch.Tensor:
        """
        Convert raw bytes to float tensor.
        Assumes 16kHz mono PCM16.
        """
        # Fixed: Input Validation (Invalid Byte Length)
        # np.frombuffer throws error if buffer size is not multiple of element size (2 for int16)
        if len(audio_data) % 2 != 0:
            logger.warning("Received odd number of bytes, trimming last byte.")
            audio_data = audio_data[:-1]

        # Convert bytes to numpy int16
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # Fixed: Input Validation (WAV Header Check)
        # Check if it's a WAV file by looking for "RIFF" header
        if len(audio_data) > 44 and audio_data[:4] == b'RIFF':
             # Simple heuristic: Skip 44 bytes (standard WAV header)
             # Better approach would be using `soundfile` or `wave` module,
             # but keeping dependencies minimal for "The Judge".
             audio_np = audio_np[22:] # 44 bytes / 2 bytes_per_sample = 22 samples

        # Convert to float32 [-1, 1]
        audio_float = audio_np.astype(np.float32) / 32768.0
        
        # Convert to tensor
        tensor = torch.from_numpy(audio_float).unsqueeze(0) # (1, T)
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
