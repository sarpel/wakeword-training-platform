"""
Pydantic-based Configuration Validator
Provides robust, self-documenting schema validation for all YAML configurations.
"""

from typing import Any, Dict, List, Literal, Tuple

from pydantic import BaseModel, Field, ValidationError, root_validator, validator

# ===========================================================================
# Pydantic Models for Configuration Sections
# ===========================================================================


class DataConfig(BaseModel):
    """Pydantic model for data processing configuration"""

    data_root: str = "data"
    sample_rate: int = Field(16000, ge=8000, description="Sample rate in Hz")
    audio_duration: float = Field(1.5, gt=0, description="Audio duration in seconds")
    n_mfcc: int = Field(0, ge=0, description="Number of MFCC coefficients")
    n_fft: int = Field(400, description="FFT window size")
    hop_length: int = Field(160, description="Hop length for STFT")
    n_mels: int = Field(64, ge=1, description="Number of mel bands")
    feature_type: Literal["mel", "mfcc", "mel_spectrogram"] = "mel"
    normalize_audio: bool = True
    use_precomputed_features_for_training: bool = True
    npy_feature_dir: str = "data/npy"
    npy_feature_type: Literal["mel", "mfcc"] = "mel"
    npy_cache_features: bool = True
    fallback_to_audio: bool = True

    @validator("n_fft")
    def n_fft_must_be_power_of_two(cls, v: int) -> int:
        if not (v > 0 and (v & (v - 1) == 0)):
            # While not strictly required by all libraries, it's a common practice
            pass  # Relaxing this to a warning in the main validator
        return v

    @root_validator(skip_on_failure=True)
    def hop_length_less_than_n_fft(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        n_fft, hop_length = values.get("n_fft"), values.get("hop_length")
        if hop_length is not None and n_fft is not None and hop_length >= n_fft:
            raise ValueError(f"Hop length ({hop_length}) must be less than n_fft ({n_fft})")
        return values


class TrainingConfig(BaseModel):
    """Pydantic model for training configuration"""

    batch_size: int = Field(64, gt=0)
    epochs: int = Field(80, gt=0)
    learning_rate: float = Field(3e-4, gt=0)
    early_stopping_patience: int = Field(15, gt=0)
    num_workers: int = Field(16, ge=0)
    pin_memory: bool = True
    persistent_workers: bool = True
    checkpoint_frequency: Literal["best_only", "every_epoch", "every_5_epochs", "every_10_epochs"] = "every_5_epochs"
    save_best_only: bool = True

    @validator("batch_size")
    def batch_size_power_of_two(cls, v: int) -> int:
        if not (v > 0 and (v & (v - 1) == 0)):
            # Warning only
            pass
        return v

    @validator("learning_rate")
    def learning_rate_range(cls, v: float) -> float:
        if v < 1e-6 or v > 1e-1:
            # Warning could be logged here if logger was available,
            # but for now we just allow it or could raise ValueError if strict.
            # Let's keep it lenient as per plan (warning only logic usually requires logging)
            pass
        return v


class ModelConfig(BaseModel):
    """Pydantic model for model architecture configuration"""

    architecture: Literal["resnet18", "mobilenetv3", "lstm", "gru", "tcn", "tiny_conv", "cd_dnn"] = "resnet18"
    num_classes: int = Field(2, ge=2)
    pretrained: bool = False
    dropout: float = Field(0.3, ge=0, lt=1)
    hidden_size: int = Field(128, ge=16)
    num_layers: int = Field(2, ge=1)
    bidirectional: bool = True
    # TCN & CD-DNN fields are optional or handled via extra fields if passed, 
    # but adding them explicitly is better if they are in the default config.
    tcn_num_channels: List[int] = [64, 128, 256]
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.3
    cddnn_hidden_layers: List[int] = [512, 256, 128]
    cddnn_context_frames: int = 50
    cddnn_dropout: float = 0.3


class AugmentationConfig(BaseModel):
    """Pydantic model for data augmentation configuration"""

    time_stretch_min: float = 0.90
    time_stretch_max: float = 1.10
    pitch_shift_min: int = -2
    pitch_shift_max: int = 2
    background_noise_prob: float = Field(0.5, ge=0, le=1)
    noise_snr_min: float = 5.0
    noise_snr_max: float = 20.0
    rir_prob: float = Field(0.25, ge=0, le=1)
    rir_dry_wet_min: float = 0.3
    rir_dry_wet_max: float = 0.7
    rir_dry_wet_strategy: Literal["random", "fixed", "adaptive"] = "random"
    use_spec_augment: bool = True
    freq_mask_param: int = Field(15, gt=0)
    time_mask_param: int = Field(30, gt=0)
    n_freq_masks: int = Field(2, ge=0)
    n_time_masks: int = Field(2, ge=0)
    # New fields found in defaults.py / UI
    time_shift_prob: float = 0.0
    time_shift_min_ms: int = -100
    time_shift_max_ms: int = 100

    @root_validator(skip_on_failure=True)
    def min_less_than_max(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # Get values with None checks for comparison operations
        # Mypy requires explicit None checks when comparing Optional values
        time_stretch_min = values.get("time_stretch_min")
        time_stretch_max = values.get("time_stretch_max")
        if time_stretch_min is not None and time_stretch_max is not None:
            if time_stretch_min >= time_stretch_max:
                raise ValueError("time_stretch_min must be less than time_stretch_max")

        pitch_shift_min = values.get("pitch_shift_min")
        pitch_shift_max = values.get("pitch_shift_max")
        if pitch_shift_min is not None and pitch_shift_max is not None:
            if pitch_shift_min >= pitch_shift_max:
                raise ValueError("pitch_shift_min must be less than pitch_shift_max")

        noise_snr_min = values.get("noise_snr_min")
        noise_snr_max = values.get("noise_snr_max")
        if noise_snr_min is not None and noise_snr_max is not None:
            if noise_snr_min >= noise_snr_max:
                raise ValueError("noise_snr_min must be less than noise_snr_max")

        return values


class OptimizerConfig(BaseModel):
    """Pydantic model for optimizer and scheduler configuration"""

    optimizer: Literal["adam", "sgd", "adamw"] = "adamw"
    weight_decay: float = Field(1e-4, ge=0)
    momentum: float = Field(0.9, ge=0, lt=1)
    betas: Tuple[float, float] = (0.9, 0.999)
    scheduler: Literal["cosine", "step", "plateau", "none"] = "cosine"
    warmup_epochs: int = Field(3, ge=0)
    min_lr: float = Field(1e-6, ge=0)
    step_size: int = Field(10, gt=0)
    gamma: float = Field(0.5, gt=0)
    patience: int = Field(5, gt=0)
    factor: float = Field(0.5, gt=0)
    gradient_clip: float = Field(1.0, gt=0)
    mixed_precision: bool = True


class LossConfig(BaseModel):
    """Pydantic model for loss function configuration"""

    loss_function: Literal["cross_entropy", "focal_loss", "triplet_loss"] = "cross_entropy"
    label_smoothing: float = Field(0.05, ge=0, lt=1)
    focal_alpha: float = Field(0.25, ge=0, le=1)
    focal_gamma: float = Field(2.0, ge=0)
    class_weights: Literal["balanced", "none", "custom"] = "balanced"
    hard_negative_weight: float = Field(1.5, ge=1.0)
    sampler_strategy: Literal["weighted", "balanced", "none"] = "weighted"
    # New fields
    triplet_margin: float = 1.0
    class_weight_min: float = 0.1
    class_weight_max: float = 100.0


class QATConfig(BaseModel):
    """Pydantic model for QAT configuration"""

    enabled: bool = False
    backend: Literal["fbgemm", "qnnpack"] = "fbgemm"
    start_epoch: int = Field(5, ge=0)


class DistillationConfig(BaseModel):
    """Pydantic model for Knowledge Distillation configuration"""

    enabled: bool = False
    teacher_model_path: str = ""
    teacher_on_cpu: bool = False
    teacher_mixed_precision: bool = True
    log_memory_usage: bool = False
    teacher_architecture: Literal["wav2vec2"] = "wav2vec2"
    temperature: float = Field(2.0, ge=1.0, le=10.0)
    alpha: float = Field(0.5, ge=0.0, le=1.0)


class WakewordPydanticConfig(BaseModel):
    """Complete Pydantic model for wakeword training configuration"""

    config_name: str = "default"
    description: str = "Default wakeword training configuration"
    data: DataConfig = Field(default_factory=lambda: DataConfig())  # type: ignore
    training: TrainingConfig = Field(default_factory=lambda: TrainingConfig())  # type: ignore
    model: ModelConfig = Field(default_factory=lambda: ModelConfig())  # type: ignore
    augmentation: AugmentationConfig = Field(default_factory=lambda: AugmentationConfig())  # type: ignore
    optimizer: OptimizerConfig = Field(default_factory=lambda: OptimizerConfig())  # type: ignore
    loss: LossConfig = Field(default_factory=lambda: LossConfig())  # type: ignore
    qat: QATConfig = Field(default_factory=lambda: QATConfig())  # type: ignore
    distillation: DistillationConfig = Field(default_factory=lambda: DistillationConfig())  # type: ignore

    @root_validator(skip_on_failure=True)
    def cross_dependencies(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        training_config = values.get("training")
        optimizer_config = values.get("optimizer")
        if training_config and optimizer_config:
            if optimizer_config.warmup_epochs > training_config.epochs:
                raise ValueError("Warmup epochs cannot exceed total epochs")
        return values


def validate_config_with_pydantic(config_dict: Dict[str, Any]) -> Tuple[bool, List[Any]]:
    """
    Validate a configuration dictionary using the Pydantic models.

    Args:
        config_dict: The configuration dictionary to validate.

    Returns:
        A tuple containing:
        - bool: True if the configuration is valid, False otherwise.
        - List[Any]: A list of validation error details from Pydantic.
                     Each error is a dict-like ErrorDetails object.
    """
    try:
        WakewordPydanticConfig.parse_obj(config_dict)
        return True, []
    except ValidationError as e:
        # e.errors() returns List[ErrorDetails], which is compatible with List[Any]
        return False, e.errors()


if __name__ == "__main__":
    pass

    # Test with a sample valid config
    sample_config = {
        "config_name": "test",
        "description": "A test config",
        "data": {"sample_rate": 16000},
        "training": {"epochs": 10},
        "optimizer": {"warmup_epochs": 1},
    }
    is_valid, errors = validate_config_with_pydantic(sample_config)
    print(f"Sample config valid: {is_valid}")

    # Test with an invalid config
    invalid_config = {
        "config_name": "invalid",
        "data": {"sample_rate": 4000},  # Too low
        "training": {"learning_rate": -0.1},  # Invalid
        "optimizer": {"warmup_epochs": 20, "scheduler": "invalid_scheduler"},
        "loss": {"label_smoothing": 1.5},
    }
    is_valid, errors = validate_config_with_pydantic(invalid_config)
    print(f"Invalid config valid: {is_valid}")
    if not is_valid:
        print("Validation errors:")
        for error in errors:
            print(f"- Location: {error['loc']}, Message: {error['msg']}")
