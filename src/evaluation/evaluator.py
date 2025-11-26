"""
Model Evaluator for File-Based and Test Set Evaluation
GPU-accelerated batch evaluation with comprehensive metrics
"""
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import time
import structlog

logger = structlog.get_logger(__name__)

class MetricResults:
    accuracy: float
    precision: float
    recall: float
    f1: float
    loss: float = 0.0
    confusion_matrix: Any = None

@dataclass
class EvaluationResult:
    """Single file evaluation result"""
    filename: str
    prediction: str  # "Positive" or "Negative"
    confidence: float
    latency_ms: float
    logits: np.ndarray


class ModelEvaluator:
    """
    Model evaluator for file-based and batch evaluation
    """

    def __init__(
        self,
        model: nn.Module,
        sample_rate: int = 16000,
        audio_duration: float = 1.5,
        device: str = 'cuda',
        feature_type: str = 'mel',
        n_mels: int = 64,
        n_mfcc: int = 0,
        n_fft: int = 400,
        hop_length: int = 160
    ):
        """
        Initialize model evaluator

        Args:
            model: Trained PyTorch model
            sample_rate: Audio sample rate
            audio_duration: Audio duration in seconds
            device: Device for inference ('cuda' or 'cpu')
            feature_type: Feature type ('mel' or 'mfcc')
            n_mels: Number of mel filterbanks
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        # Enforce CUDA
        enforce_cuda()

        self.model = model
        self.sample_rate = sample_rate
        self.audio_duration = audio_duration
        self.device = device

        # Move model to device and set to eval mode
        self.model.to(device)
        self.model.eval()

        # Audio processor
        self.audio_processor = AudioProcessor(
            target_sr=sample_rate,
            target_duration=audio_duration
        )

        # Normalize feature type (handle legacy 'mel_spectrogram')
        if feature_type == 'mel_spectrogram':
            feature_type = 'mel'

        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            sample_rate=sample_rate,
            feature_type=feature_type,
            n_mels=n_mels,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            device=device
        )

        # Metrics calculator
        self.metrics_calculator = MetricsCalculator(device=device)

        logger.info(f"ModelEvaluator initialized on {device}")

    def evaluate_file(self, audio_path: Path, threshold: float = 0.5) -> EvaluationResult:
        return evaluate_file(self, audio_path, threshold)

    def evaluate_files(self, audio_paths: List[Path], threshold: float = 0.5, batch_size: int = 32) -> List[EvaluationResult]:
        return evaluate_files(self, audio_paths, threshold, batch_size)

    def evaluate_dataset(self, dataset, threshold: float = 0.5, batch_size: int = 32) -> Tuple[MetricResults, List[EvaluationResult]]:
        return evaluate_dataset(self, dataset, threshold, batch_size)

    def get_roc_curve_data(self, dataset, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return get_roc_curve_data(self, dataset, batch_size)

    def evaluate_with_advanced_metrics(self, dataset, total_seconds: float, target_fah: float = 1.0, batch_size: int = 32) -> Dict:
        return evaluate_with_advanced_metrics(self, dataset, total_seconds, target_fah, batch_size)

    


def load_model_for_evaluation(
    checkpoint_path: Path,
    device: str = 'cuda'
) -> Tuple[nn.Module, Dict]:
    """
    Load trained model from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Tuple of (model, config_dict)
    """
    logger.info(f"Loading model from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config
    if 'config' not in checkpoint:
        raise ValueError("Checkpoint does not contain configuration")

    config_data = checkpoint['config']

    # Convert config dict to WakewordConfig object if needed
    from src.config.defaults import WakewordConfig

    if isinstance(config_data, dict):
        # Config was saved as dict, need to reconstruct
        config = WakewordConfig.from_dict(config_data)
        logger.info("Converted config dict to WakewordConfig object")
    else:
        # Config is already a WakewordConfig object
        config = config_data

    # Create model
    from src.models.architectures import create_model

    model = create_model(
        architecture=config.model.architecture,
        num_classes=config.model.num_classes,
        pretrained=False,
        dropout=config.model.dropout
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move to device and set eval mode
    model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully: {config.model.architecture}")

    # Get additional info
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'val_loss': checkpoint.get('val_loss', 0.0),
        'val_metrics': checkpoint.get('val_metrics', {}),
        'config': config
    }

    return model, info


if __name__ == "__main__":
    # Test model loading and evaluation
    print("Model Evaluator Test")
    print("=" * 60)

    checkpoint_path = Path("models/checkpoints/best_model.pt")

    if checkpoint_path.exists():
        try:
            model, info = load_model_for_evaluation(checkpoint_path)
            print(f"✅ Model loaded: {info['config'].model.architecture}")
            print(f"   Epoch: {info['epoch']}")
            print(f"   Val Loss: {info['val_loss']:.4f}")

            # Create evaluator
            evaluator = ModelEvaluator(
                model=model,
                sample_rate=info['config'].data.sample_rate,
                audio_duration=info['config'].data.audio_duration,
                feature_type=info['config'].data.feature_type,
                n_mels=info['config'].data.n_mels,
                n_mfcc=info['config'].data.n_mfcc,
                n_fft=info['config'].data.n_fft,
                hop_length=info['config'].data.hop_length
            )

            print(f"✅ Evaluator created")
            print("\nEvaluator module loaded successfully")

        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print(f"⚠️  No checkpoint found at: {checkpoint_path}")
        print("Train a model first (Panel 3)")

    print("\nEvaluation module ready")
