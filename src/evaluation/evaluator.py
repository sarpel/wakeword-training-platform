"""
Model Evaluator for File-Based and Test Set Evaluation
GPU-accelerated batch evaluation with comprehensive metrics
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union, cast

if TYPE_CHECKING:
    from src.config.defaults import WakewordConfig

import numpy as np
import structlog
import torch
import torch.nn as nn

from src.config.cuda_utils import enforce_cuda
from src.data.audio_utils import AudioProcessor as CpuAudioProcessor  # Renamed for clarity
from src.data.feature_extraction import FeatureExtractor
from src.data.processor import AudioProcessor as GpuAudioProcessor  # This is the one we want
from src.evaluation.advanced_evaluator import evaluate_with_advanced_metrics
from src.evaluation.dataset_evaluator import evaluate_dataset, get_roc_curve_data
from src.evaluation.file_evaluator import evaluate_file, evaluate_files
from src.evaluation.types import EvaluationResult
from src.training.metrics import MetricResults, MetricsCalculator

logger = structlog.get_logger(__name__)


class ModelEvaluator:
    """
    Model evaluator for file-based and batch evaluation
    """

    def __init__(
        self,
        model: nn.Module,
        sample_rate: int = 16000,
        audio_duration: float = 1.5,
        device: str = "cuda",
        feature_type: str = "mel",
        n_mels: int = 64,
        n_mfcc: int = 0,
        n_fft: int = 400,
        hop_length: int = 160,
        config: Optional[Union[Dict[str, Any], "WakewordConfig"]] = None,
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
        # Enforce CUDA (allow CPU)
        enforce_cuda(allow_cpu=True)

        self.model = model
        self.sample_rate = sample_rate
        self.audio_duration = audio_duration
        self.device = device

        # Move model to device and set to eval mode
        self.model.to(device)
        self.model.eval()

        # Audio processor
        # Create CMVN path
        from src.config.paths import paths

        cmvn_path = paths.CMVN_STATS

        if not cmvn_path.exists():
            logger.warning(
                f"CMVN stats not found at {cmvn_path}. Evaluation might be inaccurate if model was trained with CMVN."
            )

        processor_config: Optional["WakewordConfig"] = None
        if config is not None:
            from src.config.defaults import WakewordConfig

            if isinstance(config, dict):
                processor_config = WakewordConfig.from_dict(config)
            else:
                processor_config = config

        if processor_config is None:
            from src.config.defaults import WakewordConfig

            processor_config = WakewordConfig()

        self.audio_processor = GpuAudioProcessor(
            config=processor_config, cmvn_path=cmvn_path if cmvn_path.exists() else None, device=device
        )

        # CPU Audio Processor for file loading
        self.cpu_audio_processor = CpuAudioProcessor(target_sr=sample_rate, target_duration=audio_duration)

        # Normalize feature type (handle legacy 'mel_spectrogram')
        if feature_type == "mel_spectrogram":
            feature_type = "mel"

        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            sample_rate=sample_rate,
            feature_type=cast(Literal["mel", "mfcc"], feature_type),
            n_mels=n_mels,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            device=device,
        )

        # Metrics calculator
        self.metrics_calculator = MetricsCalculator(device=device)

        logger.info(f"ModelEvaluator initialized on {device}")
        logger.info(f"Class Mapping: Positive=1, Negative=0")
        if hasattr(model, "num_classes"):
            logger.info(f"Model Classes: {model.num_classes}")

    def evaluate_file(self, audio_path: Path, threshold: float = 0.5) -> EvaluationResult:
        return evaluate_file(self, audio_path, threshold)

    def evaluate_files(
        self, audio_paths: List[Path], threshold: float = 0.5, batch_size: int = 32
    ) -> List[EvaluationResult]:
        return evaluate_files(self, audio_paths, threshold, batch_size)

    def evaluate_dataset(
        self, dataset: Any, threshold: float = 0.5, batch_size: int = 32
    ) -> Tuple[MetricResults, List[EvaluationResult]]:
        return evaluate_dataset(self, dataset, threshold, batch_size)

    def get_roc_curve_data(self, dataset: Any, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return get_roc_curve_data(self, dataset, batch_size)

    def evaluate_with_advanced_metrics(
        self,
        dataset: Any,
        total_seconds: float,
        target_fah: float = 1.0,
        batch_size: int = 32,
    ) -> Dict:
        return evaluate_with_advanced_metrics(self, dataset, total_seconds, target_fah, batch_size)


def load_model_for_evaluation(checkpoint_path: Path, device: str = "cuda") -> Tuple[nn.Module, Dict]:
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
    if "config" not in checkpoint:
        raise ValueError("Checkpoint does not contain configuration")

    config_data = checkpoint["config"]

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

    # Calculate input size for model
    input_samples = int(config.data.sample_rate * config.data.audio_duration)
    time_steps = input_samples // config.data.hop_length + 1

    feature_dim = (
        config.data.n_mels
        if config.data.feature_type == "mel_spectrogram" or config.data.feature_type == "mel"
        else config.data.n_mfcc
    )

    if config.model.architecture == "cd_dnn":
        input_size = feature_dim * time_steps
    else:
        input_size = feature_dim

    model = create_model(
        architecture=config.model.architecture,
        num_classes=config.model.num_classes,
        pretrained=False,
        dropout=config.model.dropout,
        input_size=input_size,
        input_channels=1,
        # RNN (LSTM/GRU) params
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

    # Load weights
    state_dict = checkpoint["model_state_dict"]

    # --- Robust Loading Logic ---
    model_state = model.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        # Handle MobileNetV3 remapping
        # Old checkpoints might have 'mobilenet.features.X' but model expects 'features.X'
        if k.startswith("mobilenet.features.") and "features." + k[19:] in model_state:
            new_key = "features." + k[19:]
            new_state_dict[new_key] = v
        # Keep original key if it matches
        elif k in model_state:
            new_state_dict[k] = v
        else:
            # Keep it anyway for strict=False to catch, or we can try to map
            new_state_dict[k] = v

    # Filter out keys with size mismatches to prevent crashes
    final_state_dict = {}
    for k, v in new_state_dict.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                final_state_dict[k] = v
            else:
                logger.warning(f"Skipping key {k}: Shape mismatch. Cpt: {v.shape}, Mdl: {model_state[k].shape}")
        else:
            # If key not in model, we can't load it anyway
            pass

    # Handle QAT checkpoints loaded into FP32 models
    # Filter out quantization keys that are not in the model
    # (Existing logic preserved/merged)

    # Load with strict=False to allow missing unused keys (like mobilenet.classifier)
    missing, unexpected = model.load_state_dict(final_state_dict, strict=False)

    if missing:
        logger.warning(f"Missing keys (some may be expected if architecture changed): {missing[:5]}...")
    if unexpected:
        logger.info(f"Unexpected keys in checkpoint (ignored): {unexpected[:5]}...")

    # Move to device and set eval mode
    model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully: {config.model.architecture}")

    # Get additional info
    info = {
        "epoch": checkpoint.get("epoch", 0),
        "val_loss": checkpoint.get("val_loss", 0.0),
        "val_metrics": checkpoint.get("val_metrics", {}),
        "config": config,
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
                sample_rate=info["config"].data.sample_rate,
                audio_duration=info["config"].data.audio_duration,
                feature_type=info["config"].data.feature_type,
                n_mels=info["config"].data.n_mels,
                n_mfcc=info["config"].data.n_mfcc,
                n_fft=info["config"].data.n_fft,
                hop_length=info["config"].data.hop_length,
            )

            print(f"✅ Evaluator created")
            print("\nEvaluator module loaded successfully")

        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print(f"⚠️  No checkpoint found at: {checkpoint_path}")
        print("Train a model first (Panel 3)")

    print("\nEvaluation module ready")
