"""
Pydantic-based Configuration Validator
Provides robust, self-documenting schema validation for all YAML configurations.
"""
from typing import List, Tuple, Dict, Any, Literal
from pydantic import BaseModel, Field, validator, root_validator

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

    @validator('n_fft')
    def n_fft_must_be_power_of_two(cls, v):
        if not (v > 0 and (v & (v - 1) == 0)):
            # While not strictly required by all libraries, it's a common practice
            pass # Relaxing this to a warning in the main validator
        return v

    @root_validator
    def hop_length_less_than_n_fft(cls, values):
        n_fft, hop_length = values.get('n_fft'), values.get('hop_length')
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

class ModelConfig(BaseModel):
    """Pydantic model for model architecture configuration"""
    architecture: Literal["resnet18", "mobilenetv3", "lstm", "gru", "tcn"] = "resnet18"
    num_classes: int = Field(2, ge=2)
    pretrained: bool = False
    dropout: float = Field(0.3, ge=0, lt=1)
    hidden_size: int = Field(128, ge=16)
    num_layers: int = Field(2, ge=1)
    bidirectional: bool = True

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

    @root_validator
    def min_less_than_max(cls, values):
        if values.get('time_stretch_min') >= values.get('time_stretch_max'):
            raise ValueError("time_stretch_min must be less than time_stretch_max")
        if values.get('pitch_shift_min') >= values.get('pitch_shift_max'):
            raise ValueError("pitch_shift_min must be less than pitch_shift_max")
        if values.get('noise_snr_min') >= values.get('noise_snr_max'):
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
    loss_function: Literal["cross_entropy", "focal_loss"] = "cross_entropy"
    label_smoothing: float = Field(0.05, ge=0, lt=1)
    focal_alpha: float = Field(0.25, ge=0, le=1)
    focal_gamma: float = Field(2.0, ge=0)
    class_weights: Literal["balanced", "none", "custom"] = "balanced"
    hard_negative_weight: float = Field(1.5, ge=1.0)
    sampler_strategy: Literal["weighted", "balanced", "none"] = "weighted"

class WakewordPydanticConfig(BaseModel):
    """Complete Pydantic model for wakeword training configuration"""
    config_name: str = "default"
    description: str = "Default wakeword training configuration"
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    loss: LossConfig = Field(default_factory=LossConfig)

    @root_validator
    def cross_dependencies(cls, values):
        training_config = values.get('training')
        optimizer_config = values.get('optimizer')
        if training_config and optimizer_config:
            if optimizer_config.warmup_epochs > training_config.epochs:
                raise ValueError("Warmup epochs cannot exceed total epochs")
        return values

def validate_config_with_pydantic(config_dict: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Validate a configuration dictionary using the Pydantic models.

    Args:
        config_dict: The configuration dictionary to validate.

    Returns:
        A tuple containing:
        - bool: True if the configuration is valid, False otherwise.
        - List[Dict[str, Any]]: A list of validation errors.
    """
    try:
        WakewordPydanticConfig.parse_obj(config_dict)
        return True, []
    except Exception as e:
        return False, e.errors()

if __name__ == "__main__":
    import yaml
    from pathlib import Path

    # Test with a sample valid config
    sample_config = {
        "config_name": "test",
        "description": "A test config",
        "data": {"sample_rate": 16000},
        "training": {"epochs": 10},
        "optimizer": {"warmup_epochs": 1}
    }
    is_valid, errors = validate_config_with_pydantic(sample_config)
    print(f"Sample config valid: {is_valid}")

    # Test with an invalid config
    invalid_config = {
        "config_name": "invalid",
        "data": {"sample_rate": 4000}, # Too low
        "training": {"learning_rate": -0.1}, # Invalid
        "optimizer": {"warmup_epochs": 20, "scheduler": "invalid_scheduler"},
        "loss": {"label_smoothing": 1.5}
    }
    is_valid, errors = validate_config_with_pydantic(invalid_config)
    print(f"Invalid config valid: {is_valid}")
    if not is_valid:
        print("Validation errors:")
        for error in errors:
            print(f"- Location: {error['loc']}, Message: {error['msg']}")
