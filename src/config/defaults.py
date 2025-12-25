"""
Default Configuration Parameters for Wakeword Training
Defines basic and advanced training hyperparameters
"""

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from src.config.env_config import env_config


@dataclass
class DataConfig:
    """Data processing configuration"""

    data_root: str = "data"
    # Audio parameters
    sample_rate: int = 16000  # Hz
    audio_duration: float = 1.5  # seconds
    n_mfcc: int = 0  # Number of MFCC coefficients
    n_fft: int = 400  # FFT window size
    hop_length: int = 160  # Hop length for STFT
    n_mels: int = 64  # Number of mel bands

    # Feature extraction
    feature_type: str = "mel"  # mel, mfcc
    normalize_audio: bool = True

    # NEW: NPY feature parameters
    use_precomputed_features_for_training: bool = True  # Enable NPY loading
    npy_feature_dir: str = "data/npy"  # Directory with split .npy files (train/val/test)
    npy_feature_type: str = "mel"  # mel, mfcc (must match extraction)
    npy_cache_features: bool = True  # Cache loaded features in RAM
    fallback_to_audio: bool = True  # If NPY missing, load raw audio

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class TrainingConfig:
    """Training configuration"""

    # Basic training parameters
    batch_size: int = 64
    epochs: int = 80
    learning_rate: float = 5e-4  # Optimized: 5e-4 for stable training
    early_stopping_patience: int = 15

    # FNR target for early stopping (None = use F1-based stopping)
    fnr_target: Optional[float] = None  # Target FNR (e.g., 0.02 for 2%)

    # Hardware
    num_workers: int = env_config.get_int("TRAINING_NUM_WORKERS", 8)  # Dynamic default
    pin_memory: bool = True
    persistent_workers: bool = True
    use_compile: bool = env_config.use_triton  # Enable torch.compile if Triton supported
    use_gradient_checkpointing: bool = False  # VRAM optimization

    # Checkpointing
    checkpoint_frequency: str = "every_5_epochs"  # best_only, every_epoch, every_5_epochs, every_10_epochs
    save_best_only: bool = True

    # EMA Parameters
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_final_decay: float = 0.9995
    ema_final_epochs: int = 10

    # Metrics
    metric_window_size: int = 100

    # Hard Negative Mining
    include_mined_negatives: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ModelConfig:
    """Model architecture configuration"""

    architecture: str = "resnet18"  # resnet18, mobilenetv3, lstm, gru, tcn, tiny_conv
    num_classes: int = 2  # Binary classification
    pretrained: bool = False  # Use pretrained weights (ImageNet)
    dropout: float = 0.3

    # Architecture-specific parameters
    hidden_size: int = 128  # For LSTM/GRU
    num_layers: int = 2  # For LSTM/GRU
    bidirectional: bool = True  # For LSTM/GRU

    # TCN Parameters
    tcn_num_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.3

    # CD-DNN Parameters
    cddnn_hidden_layers: List[int] = field(default_factory=lambda: [512, 256, 128])
    cddnn_context_frames: int = 50
    cddnn_dropout: float = 0.3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class AugmentationConfig:
    """Data augmentation configuration"""

    # Time domain augmentation (CPU-based)
    time_stretch_min: float = 0.90
    time_stretch_max: float = 1.10
    pitch_shift_min: int = -2  # semitones
    pitch_shift_max: int = 2  # semitones

    # Time shift (New)
    time_shift_prob: float = 0.0
    time_shift_min_ms: int = -100
    time_shift_max_ms: int = 100

    # Noise augmentation (CPU-based)
    background_noise_prob: float = 0.5
    noise_snr_min: float = 5.0  # dB
    noise_snr_max: float = 20.0  # dB

    # RIR augmentation (CPU-based)
    rir_prob: float = 0.25

    # NEW: RIR dry/wet mixing parameters
    rir_dry_wet_min: float = 0.3  # Minimum dry ratio (30% dry, 70% wet)
    rir_dry_wet_max: float = 0.7  # Maximum dry ratio (70% dry, 30% wet)
    rir_dry_wet_strategy: str = "random"  # random, fixed, adaptive

    # SpecAugment (GPU-based, applied in trainer)
    use_spec_augment: bool = True
    freq_mask_param: int = 15
    time_mask_param: int = 30
    n_freq_masks: int = 2
    n_time_masks: int = 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class OptimizerConfig:
    """Optimizer and scheduler configuration"""

    optimizer: str = "adamw"  # adam, sgd, adamw
    weight_decay: float = 1e-4
    momentum: float = 0.9  # For SGD
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])

    # Scheduler
    scheduler: str = "cosine"  # cosine, step, plateau, none
    warmup_epochs: int = 5  # Optimized: 5 epochs warmup for stable training
    min_lr: float = 1e-6

    # Step scheduler parameters
    step_size: int = 10
    gamma: float = 0.5

    # Plateau scheduler parameters
    patience: int = 5
    factor: float = 0.5

    # Gradient
    gradient_clip: float = 1.0
    mixed_precision: bool = True  # Default True for RTX 3060 and better performance

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class LossConfig:
    """Loss function configuration"""

    loss_function: str = "focal_loss"  # Changed to focal_loss for FNR optimization
    label_smoothing: float = 0.1  # Increased label smoothing

    # Focal loss parameters - FNR focused!
    focal_alpha: float = 0.85  # FNR oriented (0.85-0.90 range)
    focal_gamma: float = 2.5  # Increased for hard example mining

    # Triplet loss parameters
    triplet_margin: float = 1.0

    # Class weighting
    class_weights: str = "balanced"  # balanced, none, custom
    hard_negative_weight: float = 2.0  # Increased for FNR optimization
    class_weight_min: float = 0.1
    class_weight_max: float = 100.0

    # Dynamic alpha (FNR-focused training)
    use_dynamic_alpha: bool = True  # Enable dynamic alpha during training
    max_focal_alpha: float = 0.90  # Maximum alpha value for dynamic scaling

    # Sampling strategy
    sampler_strategy: str = "weighted"  # weighted, balanced, none

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class QATConfig:
    """Quantization Aware Training configuration"""

    enabled: bool = False
    backend: str = env_config.quantization_backend  # Dynamic default: fbgemm (x86), qnnpack (ARM)

    # When to start QAT (usually after some epochs of normal training)
    start_epoch: int = 5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class DistillationConfig:
    """Knowledge Distillation configuration"""

    enabled: bool = False
    teacher_model_path: str = ""

    # Memory optimization options
    teacher_on_cpu: bool = False
    teacher_mixed_precision: bool = True
    log_memory_usage: bool = False

    # Distillation parameters
    teacher_architecture: str = "dual"  # wav2vec2, conformer, dual (recommended)
    secondary_teacher_architecture: str = "conformer"
    secondary_teacher_model_path: str = ""

    temperature: float = 2.0
    temperature_scheduler: str = "fixed"  # fixed, linear_decay, exponential_decay
    alpha: float = 0.3  # Optimized: 0.3 is more balanced than 0.5

    # Feature Alignment (Intermediate Matching)
    feature_alignment_enabled: bool = False
    feature_alignment_weight: float = 0.1
    # Indices of layers to match (implementation dependent)
    alignment_layers: List[int] = field(default_factory=lambda: [1, 2, 3])

    def __post_init__(self) -> None:
        """Validate parameters after initialization"""
        if not isinstance(self.temperature, (int, float)):
            raise TypeError(f"temperature must be numeric, got {type(self.temperature)}")
        if not 1.0 <= self.temperature <= 10.0:
            raise ValueError(
                f"temperature must be in range [1.0, 10.0], got {self.temperature}. "
                f"Higher values soften distributions more."
            )
        self.temperature = float(self.temperature)

        if not isinstance(self.alpha, (int, float)):
            raise TypeError(f"alpha must be numeric, got {type(self.alpha)}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(
                f"alpha must be in range [0.0, 1.0], got {self.alpha}. "
                f"Alpha=0.0 means no distillation, alpha=1.0 means ignore ground truth."
            )
        self.alpha = float(self.alpha)

        valid_architectures = ["wav2vec2", "conformer", "dual"]
        if self.teacher_architecture not in valid_architectures:
            raise ValueError(
                f"teacher_architecture must be one of {valid_architectures}, got '{self.teacher_architecture}'"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class CMVNConfig:
    """CMVN configuration"""

    enabled: bool = True
    stats_path: str = "data/cache/cmvn_stats.json"
    calculate_on_fly: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class StreamingConfig:
    """Streaming detection configuration"""

    hysteresis_high: float = 0.7
    hysteresis_low: float = 0.3
    buffer_length_ms: int = 1500
    smoothing_window: int = 5
    cooldown_ms: int = 500

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class SizeTargetConfig:
    """Model size target configuration"""

    max_flash_kb: int = 0  # 0 means no limit
    max_ram_kb: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class CalibrationConfig:
    """Quantization calibration configuration"""

    num_samples: int = 100
    positive_ratio: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class WakewordConfig:
    """Complete wakeword training configuration"""

    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    # New optional configurations
    qat: QATConfig = field(default_factory=QATConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    cmvn: CMVNConfig = field(default_factory=CMVNConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    size_targets: SizeTargetConfig = field(default_factory=SizeTargetConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    # Metadata
    config_name: str = "default"
    description: str = "Default wakeword training configuration"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "config_name": self.config_name,
            "description": self.description,
            "data": self.data.to_dict(),
            "training": self.training.to_dict(),
            "model": self.model.to_dict(),
            "augmentation": self.augmentation.to_dict(),
            "optimizer": self.optimizer.to_dict(),
            "loss": self.loss.to_dict(),
            "qat": self.qat.to_dict(),
            "distillation": self.distillation.to_dict(),
            "cmvn": self.cmvn.to_dict(),
            "streaming": self.streaming.to_dict(),
            "size_targets": self.size_targets.to_dict(),
            "calibration": self.calibration.to_dict(),
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "WakewordConfig":
        """Create configuration from dictionary"""
        return cls(
            config_name=config_dict.get("config_name", "default"),
            description=config_dict.get("description", ""),
            data=DataConfig(**config_dict.get("data", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            augmentation=AugmentationConfig(**config_dict.get("augmentation", {})),
            optimizer=OptimizerConfig(**config_dict.get("optimizer", {})),
            loss=LossConfig(**config_dict.get("loss", {})),
            qat=QATConfig(**config_dict.get("qat", {})),
            distillation=DistillationConfig(**config_dict.get("distillation", {})),
            cmvn=CMVNConfig(**config_dict.get("cmvn", {})),
            streaming=StreamingConfig(**config_dict.get("streaming", {})),
            size_targets=SizeTargetConfig(**config_dict.get("size_targets", {})),
            calibration=CalibrationConfig(**config_dict.get("calibration", {})),
        )

    def save(self, path: Path) -> None:
        """Save configuration to YAML file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)  # <-- safe_dump

    @classmethod
    def load(cls, path: Path) -> "WakewordConfig":
        """Load configuration from YAML file"""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    def copy(self) -> "WakewordConfig":
        """Create a deep copy of the configuration"""
        return copy.deepcopy(self)


def get_default_config() -> WakewordConfig:
    """
    Get default configuration

    Returns:
        Default WakewordConfig instance
    """
    return WakewordConfig(
        config_name="default",
        description="Default balanced configuration for general use",
    )


def load_latest_hpo_profile(config: WakewordConfig, profile_dir: Optional[Path] = None) -> bool:
    """
    Load the latest complete HPO profile from disk into an existing config instance.

    Args:
        config: The WakewordConfig instance to update
        profile_dir: Optional custom directory to search for profiles

    Returns:
        bool: True if profile was successfully loaded, False otherwise
    """
    if profile_dir is None:
        profile_dir = Path("configs/profiles")
    else:
        profile_dir = Path(profile_dir)

    profile_path = profile_dir / "hpo_best_complete.json"

    if not profile_path.exists():
        return False

    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)

        if "parameters" not in profile_data:
            return False

        params = profile_data["parameters"]

        # Update config sections
        if "training" in params:
            for k, v in params["training"].items():
                if hasattr(config.training, k):
                    setattr(config.training, k, v)

        if "optimizer" in params:
            for k, v in params["optimizer"].items():
                if hasattr(config.optimizer, k):
                    setattr(config.optimizer, k, v)

        if "model" in params:
            for k, v in params["model"].items():
                if hasattr(config.model, k):
                    setattr(config.model, k, v)

        if "augmentation" in params:
            for k, v in params["augmentation"].items():
                if hasattr(config.augmentation, k):
                    setattr(config.augmentation, k, v)

        if "loss" in params:
            for k, v in params["loss"].items():
                if hasattr(config.loss, k):
                    setattr(config.loss, k, v)

        return True
    except FileNotFoundError:
        # Silently ignore if profile file doesn't exist
        return False
    except json.JSONDecodeError as e:
        print(f"Error decoding HPO profile JSON: {e}")
        return False
    except KeyError as e:
        print(f"Error accessing HPO profile data - missing key: {e}")
        return False


# Export all configurations
__all__ = [
    "DataConfig",
    "TrainingConfig",
    "ModelConfig",
    "AugmentationConfig",
    "OptimizerConfig",
    "LossConfig",
    "QATConfig",
    "DistillationConfig",
    "WakewordConfig",
    "get_default_config",
]


if __name__ == "__main__":
    # Test configuration system
    print("Configuration System Test")
    print("=" * 60)

    # Create default config
    config = get_default_config()
    print("\nDefault Configuration:")
    print(f"  Sample Rate: {config.data.sample_rate} Hz")
    print(f"  Batch Size: {config.training.batch_size}")
    print(f"  Learning Rate: {config.training.learning_rate}")
    print(f"  Model: {config.model.architecture}")

    # Test save/load
    test_path = Path("test_config.yaml")
    config.save(test_path)
    print(f"\nConfiguration saved to: {test_path}")

    # Load back
    loaded_config = WakewordConfig.load(test_path)
    print("Configuration loaded successfully")
    print(f"  Loaded config name: {loaded_config.config_name}")

    # Cleanup
    test_path.unlink()
    print("Test file cleaned up")

    print("\nConfiguration system test complete")
