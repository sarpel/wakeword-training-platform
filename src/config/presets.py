"""
Configuration Presets for Different Use Cases
Provides optimized configurations for various scenarios
"""
from typing import Dict
from src.config.defaults import (
    WakewordConfig,
    DataConfig,
    TrainingConfig,
    ModelConfig,
    AugmentationConfig,
    OptimizerConfig,
    LossConfig
)


def get_default_preset() -> WakewordConfig:
    """
    Default balanced configuration

    Use for: General purpose, balanced performance
    """
    return WakewordConfig(
        config_name="default",
        description="Default balanced configuration for general use",
        data=DataConfig(),
        training=TrainingConfig(),
        model=ModelConfig(),
        augmentation=AugmentationConfig(),
        optimizer=OptimizerConfig(),
        loss=LossConfig()
    )


def get_small_dataset_preset() -> WakewordConfig:
    """
    Small dataset configuration (<10k samples)

    Optimized for:
    - Limited data
    - Aggressive augmentation
    - Strong regularization
    - Smaller model to avoid overfitting
    """
    return WakewordConfig(
        config_name="small_dataset",
        description="Optimized for small datasets (<10k samples) with aggressive augmentation",
        data=DataConfig(
            sample_rate=16000,
            audio_duration=1.5,
            n_mfcc=0,
            n_fft=400,
            n_mels=64
        ),
        training=TrainingConfig(
            batch_size=16,  # Smaller batch for limited data
            epochs=100,  # More epochs
            learning_rate=0.0005,  # Lower LR
            early_stopping_patience=15,
            num_workers=4
        ),
        model=ModelConfig(
            architecture="mobilenetv3",  # Smaller model to avoid overfitting
            num_classes=2,
            pretrained=False,
            dropout=0.5  # Higher dropout for regularization
        ),
        augmentation=AugmentationConfig(
            time_stretch_min=0.70,  # More aggressive stretching
            time_stretch_max=1.30,
            pitch_shift_min=-3,
            pitch_shift_max=3,
            background_noise_prob=0.7,  # High probability for data variety
            noise_snr_min=3.0,  # Lower SNR for harder examples
            noise_snr_max=25.0,
            rir_prob=0.5,  # Higher RIR usage for acoustic variety
            rir_dry_wet_min=0.2,  # More aggressive reverb range
            rir_dry_wet_max=0.8
        ),
        optimizer=OptimizerConfig(
            optimizer="adam",
            weight_decay=1e-3,  # Stronger regularization
            scheduler="cosine",
            gradient_clip=1.0,
            mixed_precision=True
        ),
        loss=LossConfig(
            loss_function="focal_loss",  # Better for imbalanced small datasets
            label_smoothing=0.0,  # Not used with focal_loss
            class_weights="balanced",
            hard_negative_weight=3.0  # Strong emphasis on hard negatives
        )
    )


def get_large_dataset_preset() -> WakewordConfig:
    """
    Large dataset configuration (>100k samples)

    Optimized for:
    - Large scale data
    - Faster training
    - Larger model capacity
    - Less aggressive augmentation
    """
    return WakewordConfig(
        config_name="large_dataset",
        description="Optimized for large datasets (>100k samples) with faster training",
        data=DataConfig(
            sample_rate=16000,
            audio_duration=1.5,
            n_mfcc=0,
            n_fft=400,
            n_mels=64
        ),
        training=TrainingConfig(
            batch_size=128,  # Larger batch for better GPU utilization
            epochs=30,  # Fewer epochs needed with large data
            learning_rate=0.002,  # Higher LR for faster convergence
            early_stopping_patience=8,
            num_workers=16  # More workers for throughput
        ),
        model=ModelConfig(
            architecture="resnet18",  # Larger model capacity
            num_classes=2,
            pretrained=False,
            dropout=0.2  # Less dropout - less overfitting risk with large data
        ),
        augmentation=AugmentationConfig(
            time_stretch_min=0.85,  # Less aggressive - data already diverse
            time_stretch_max=1.15,
            pitch_shift_min=-2,
            pitch_shift_max=2,
            background_noise_prob=0.4,  # Lower probability
            noise_snr_min=10.0,  # Higher SNR - cleaner augmentation
            noise_snr_max=20.0,
            rir_prob=0.25,
            rir_dry_wet_min=0.4,  # Lighter reverb
            rir_dry_wet_max=0.6
        ),
        optimizer=OptimizerConfig(
            optimizer="adamw",
            weight_decay=1e-4,
            scheduler="cosine",
            warmup_epochs=3,
            gradient_clip=1.0,
            mixed_precision=True
        ),
        loss=LossConfig(
            loss_function="cross_entropy",
            label_smoothing=0.05,
            class_weights="balanced",
            hard_negative_weight=2.0  # Moderate emphasis
        )
    )


def get_fast_training_preset() -> WakewordConfig:
    """
    Fast training configuration

    Optimized for:
    - Quick iteration
    - Rapid prototyping
    - Minimal augmentation
    - Faster convergence
    """
    return WakewordConfig(
        config_name="fast_training",
        description="Fast training for quick iteration and prototyping",
        data=DataConfig(
            sample_rate=16000,
            audio_duration=1.0,  # Shorter audio for speed
            n_mfcc=0,
            n_fft=400,
            n_mels=64
        ),
        training=TrainingConfig(
            batch_size=64,
            epochs=20,  # Fewer epochs for rapid iteration
            learning_rate=0.003,  # Higher LR for faster convergence
            early_stopping_patience=5,
            num_workers=6
        ),
        model=ModelConfig(
            architecture="mobilenetv3",  # Fast, lightweight architecture
            num_classes=2,
            pretrained=False,
            dropout=0.3
        ),
        augmentation=AugmentationConfig(
            time_stretch_min=0.95,  # Minimal augmentation for speed
            time_stretch_max=1.05,
            pitch_shift_min=-1,
            pitch_shift_max=1,
            background_noise_prob=0.3,  # Low probability
            noise_snr_min=15.0,  # High SNR - easy augmentation
            noise_snr_max=20.0,
            rir_prob=0.1,  # Minimal RIR usage
            rir_dry_wet_min=0.5,  # Light reverb only
            rir_dry_wet_max=0.7
        ),
        optimizer=OptimizerConfig(
            optimizer="adam",
            weight_decay=1e-4,
            scheduler="step",
            gradient_clip=1.0,
            mixed_precision=True
        ),
        loss=LossConfig(
            loss_function="cross_entropy",
            label_smoothing=0.05,
            class_weights="balanced",
            hard_negative_weight=1.5
        )
    )


def get_high_accuracy_preset() -> WakewordConfig:
    """
    High accuracy configuration

    Optimized for:
    - Maximum accuracy
    - Lower false positive rate
    - More training time
    - Comprehensive augmentation
    """
    return WakewordConfig(
        config_name="high_accuracy",
        description="Maximum accuracy with lower false positive rate",
        data=DataConfig(
            sample_rate=16000,
            audio_duration=2.0,  # Longer context for better accuracy
            n_mfcc=0,
            n_fft=400,
            n_mels=64
        ),
        training=TrainingConfig(
            batch_size=24,  # Smaller batch for better generalization
            epochs=100,  # More training time
            learning_rate=0.0005,  # Lower LR for stability
            early_stopping_patience=20,
            num_workers=4
        ),
        model=ModelConfig(
            architecture="resnet18",  # Strong, accurate architecture
            num_classes=2,
            pretrained=False,
            dropout=0.4  # Balanced dropout
        ),
        augmentation=AugmentationConfig(
            time_stretch_min=0.75,  # Wide range for robustness
            time_stretch_max=1.25,
            pitch_shift_min=-3,
            pitch_shift_max=3,
            background_noise_prob=0.6,  # High augmentation for robustness
            noise_snr_min=5.0,  # Wide SNR range
            noise_snr_max=25.0,
            rir_prob=0.4,  # High RIR for acoustic variety
            rir_dry_wet_min=0.2,  # Wide reverb range
            rir_dry_wet_max=0.8
        ),
        optimizer=OptimizerConfig(
            optimizer="adamw",
            weight_decay=5e-4,  # Strong regularization
            scheduler="cosine",
            warmup_epochs=10,
            gradient_clip=0.5,  # Stricter clipping for stability
            mixed_precision=True
        ),
        loss=LossConfig(
            loss_function="focal_loss",  # Better for hard examples
            label_smoothing=0.0,  # Not used with focal_loss
            focal_alpha=0.25,
            focal_gamma=2.5,  # Higher gamma for hard examples
            class_weights="balanced",
            hard_negative_weight=3.0,  # Strong emphasis on hard negatives
            sampler_strategy="weighted"
        )
    )


def get_edge_deployment_preset() -> WakewordConfig:
    """
    Edge deployment configuration

    Optimized for:
    - Small model size
    - Fast inference
    - Low memory footprint
    - Mobile/IoT deployment
    """
    return WakewordConfig(
        config_name="edge_deployment",
        description="Optimized for edge devices (mobile/IoT) with small model size",
        data=DataConfig(
            sample_rate=16000,
            audio_duration=1.2,  # Shorter for faster inference
            n_mfcc=0,
            n_fft=400,
            n_mels=64
        ),
        training=TrainingConfig(
            batch_size=48,
            epochs=60,
            learning_rate=0.001,
            early_stopping_patience=12,
            num_workers=4
        ),
        model=ModelConfig(
            architecture="mobilenetv3",  # Lightweight for edge devices
            num_classes=2,
            pretrained=False,
            dropout=0.25  # Lower dropout for better inference accuracy
        ),
        augmentation=AugmentationConfig(
            time_stretch_min=0.85,  # Moderate augmentation for robustness
            time_stretch_max=1.15,
            pitch_shift_min=-2,
            pitch_shift_max=2,
            background_noise_prob=0.5,  # Real-world noise variety
            noise_snr_min=8.0,
            noise_snr_max=20.0,
            rir_prob=0.3,  # Real-world acoustics
            rir_dry_wet_min=0.4,
            rir_dry_wet_max=0.6
        ),
        optimizer=OptimizerConfig(
            optimizer="adam",
            weight_decay=1e-4,
            scheduler="cosine",
            gradient_clip=1.0,
            mixed_precision=True  # Critical for edge deployment
        ),
        loss=LossConfig(
            loss_function="cross_entropy",
            label_smoothing=0.08,  # Slight smoothing for generalization
            class_weights="balanced",
            hard_negative_weight=2.5  # Reduce false positives
        )
    )


# Preset registry
PRESETS: Dict[str, callable] = {
    "Default": get_default_preset,
    "Small Dataset (<10k samples)": get_small_dataset_preset,
    "Large Dataset (>100k samples)": get_large_dataset_preset,
    "Fast Training": get_fast_training_preset,
    "High Accuracy": get_high_accuracy_preset,
    "Edge Deployment": get_edge_deployment_preset
}


def get_preset(preset_name: str) -> WakewordConfig:
    """
    Get configuration preset by name

    Args:
        preset_name: Name of preset

    Returns:
        WakewordConfig for the preset

    Raises:
        ValueError: If preset name not found
    """
    if preset_name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(
            f"Preset '{preset_name}' not found. "
            f"Available presets: {available}"
        )

    return PRESETS[preset_name]()


def list_presets() -> list:
    """
    List available preset names

    Returns:
        List of preset names
    """
    return list(PRESETS.keys())


if __name__ == "__main__":
    # Test presets
    print("Configuration Presets Test")
    print("=" * 60)

    print("\nAvailable Presets:")
    for preset_name in list_presets():
        print(f"  • {preset_name}")

    # Test each preset
    print("\nTesting Presets:")
    for preset_name in list_presets():
        config = get_preset(preset_name)
        print(f"\n{preset_name}:")
        print(f"  Description: {config.description}")
        print(f"  Model: {config.model.architecture}")
        print(f"  Batch Size: {config.training.batch_size}")
        print(f"  Epochs: {config.training.epochs}")
        print(f"  Learning Rate: {config.training.learning_rate}")

    print("\n✅ All presets loaded successfully")
    print("\nPresets test complete")