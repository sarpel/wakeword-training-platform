"""
Configuration Presets for Different Use Cases
Provides optimized configurations for various scenarios
"""

from typing import Callable, Dict

from src.config.defaults import (
    AugmentationConfig,
    CalibrationConfig,
    CMVNConfig,
    DataConfig,
    DistillationConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    QATConfig,
    SizeTargetConfig,
    StreamingConfig,
    TrainingConfig,
    WakewordConfig,
)


def get_default_preset() -> WakewordConfig:
    """
    Default balanced configuration
    """
    return WakewordConfig(
        config_name="default",
        description="Default balanced configuration for general use",
        data=DataConfig(),
        training=TrainingConfig(),
        model=ModelConfig(),
        augmentation=AugmentationConfig(
            time_shift_prob=0.3,
            time_shift_min_ms=-30,
            time_shift_max_ms=30,
        ),
        optimizer=OptimizerConfig(
            optimizer="adamw",
            weight_decay=0.01,
        ),
        loss=LossConfig(),
        cmvn=CMVNConfig(enabled=True),
        streaming=StreamingConfig(),
        size_targets=SizeTargetConfig(),
        calibration=CalibrationConfig(),
    )


def get_ha_wyoming_preset() -> WakewordConfig:
    """
    Home Assistant / Wyoming Standard configuration

    Optimized for:
    - Official Home Assistant Wyoming protocol
    - MobileNetV3-Small architecture
    - 40 Mel bands (Industry standard)
    - 1.5s window
    """
    return WakewordConfig(
        config_name="ha_wyoming",
        description="Production: Home Assistant / Wyoming Standard (MobileNetV3, 40 Mel)",
        data=DataConfig(
            sample_rate=16000,
            audio_duration=1.5,
            n_mels=40,
            hop_length=160,
        ),
        training=TrainingConfig(
            batch_size=64,
            epochs=100,
            learning_rate=0.001,
            early_stopping_patience=15,
        ),
        model=ModelConfig(
            architecture="mobilenetv3",
            num_classes=2,
            dropout=0.2,
        ),
        augmentation=AugmentationConfig(
            background_noise_prob=0.5,
            rir_prob=0.3,
            time_shift_prob=0.4,
        ),
        optimizer=OptimizerConfig(
            optimizer="adamw",
            weight_decay=0.01,
            mixed_precision=True,
        ),
        loss=LossConfig(
            loss_function="cross_entropy",
            label_smoothing=0.05,
            class_weights="balanced",
            hard_negative_weight=2.5,
        ),
        qat=QATConfig(
            enabled=True,
            backend="qnnpack",
            start_epoch=10,
        ),
        cmvn=CMVNConfig(enabled=True),
        streaming=StreamingConfig(
            hysteresis_high=0.7,
            hysteresis_low=0.3,
            buffer_length_ms=1500,
            smoothing_window=5,
        ),
        size_targets=SizeTargetConfig(
            max_flash_kb=512,
            max_ram_kb=256,
        ),
        calibration=CalibrationConfig(num_samples=200, positive_ratio=0.5),
    )


def get_esp32s3_production_preset() -> WakewordConfig:
    """
    ESP32-S3 Production configuration (PSRAM Required)

    Optimized for:
    - ESP32-S3 with PSRAM
    - MobileNetV3-Small (Int8 Quantized)
    - High accuracy edge detection
    """
    return WakewordConfig(
        config_name="esp32s3_production",
        description="Production: ESP32-S3 (PSRAM) - MobileNetV3",
        data=DataConfig(
            sample_rate=16000,
            audio_duration=1.0,
            n_mels=40,
            hop_length=160,
        ),
        training=TrainingConfig(
            batch_size=64,
            epochs=150,
            learning_rate=0.0008,
            early_stopping_patience=20,
        ),
        model=ModelConfig(
            architecture="mobilenetv3",
            num_classes=2,
            dropout=0.25,
        ),
        augmentation=AugmentationConfig(
            background_noise_prob=0.6,
            rir_prob=0.4,
            time_shift_prob=0.5,
        ),
        optimizer=OptimizerConfig(
            optimizer="adamw",
            weight_decay=0.01,
            mixed_precision=True,
        ),
        loss=LossConfig(
            loss_function="focal_loss",
            class_weights="balanced",
            hard_negative_weight=3.0,
        ),
        qat=QATConfig(
            enabled=True,
            backend="qnnpack",
            start_epoch=5,
        ),
        cmvn=CMVNConfig(enabled=True),
        streaming=StreamingConfig(
            hysteresis_high=0.75,
            hysteresis_low=0.35,
            buffer_length_ms=1000,
            smoothing_window=4,
        ),
        size_targets=SizeTargetConfig(
            max_flash_kb=1024,
            max_ram_kb=512,
        ),
        calibration=CalibrationConfig(num_samples=250, positive_ratio=0.4),
    )


def get_mcu_tiny_production_preset() -> WakewordConfig:
    """
    MCU Production configuration (No-PSRAM)

    Optimized for:
    - ESP32-S3 (Atom Echo) without PSRAM
    - TinyConv [64, 128, 128, 128] (Optimized for S3 Memory)
    - 64 Mel bands for higher resolution
    - Knowledge Distillation enabled by default
    - <250KB Peak RAM
    """
    return WakewordConfig(
        config_name="mcu_tiny_production",
        description="Production: ESP32-S3 (Atom Echo) - TinyConv High-Quality",
        data=DataConfig(
            sample_rate=16000,
            audio_duration=1.0,
            n_mels=64,             # ⬆️ Increased resolution (40 -> 64)
            hop_length=160,
        ),
        training=TrainingConfig(
            batch_size=64,
            epochs=200,
            learning_rate=0.0005,
            early_stopping_patience=25,
            use_ema=True,          # ⬆️ Enabled EMA for better stability
        ),
        model=ModelConfig(
            architecture="tiny_conv",
            num_classes=2,
            dropout=0.2,           # ⬆️ Slightly increased dropout for larger model
            tcn_num_channels=[64, 128, 128, 128], # ⬆️ Increased capacity
        ),
        augmentation=AugmentationConfig(
            background_noise_prob=0.8,
            rir_prob=0.6,          # ⬆️ Increased RIR for better robustness
            time_shift_prob=0.7,
            noise_snr_min=3,       # ⬆️ Tougher noise cases
            noise_snr_max=15,
        ),
        optimizer=OptimizerConfig(
            optimizer="adamw",
            weight_decay=0.01,     # ⬆️ Standard weight decay
            mixed_precision=True,
        ),
        loss=LossConfig(
            loss_function="focal_loss", # ⬆️ Switch to focal loss for better hard-negative handling
            label_smoothing=0.05,
            class_weights="balanced",
            hard_negative_weight=5.0,
        ),
        qat=QATConfig(
            enabled=True,
            backend="qnnpack",
            start_epoch=15,
        ),
        distillation=DistillationConfig(
            enabled=True,          # ⬆️ Enabled Knowledge Distillation
            teacher_architecture="wav2vec2",
            temperature=2.0,
            alpha=0.5,
        ),
        cmvn=CMVNConfig(enabled=True),
        streaming=StreamingConfig(
            hysteresis_high=0.85,
            hysteresis_low=0.35,
            buffer_length_ms=1000,
            smoothing_window=3,
        ),
        size_targets=SizeTargetConfig(
            max_flash_kb=256,      # ⬆️ Updated for ESP32-S3
            max_ram_kb=192,        # ⬆️ Updated for ESP32-S3 internal RAM limits
        ),
        calibration=CalibrationConfig(num_samples=300, positive_ratio=0.3),
    )


def get_server_judge_preset() -> WakewordConfig:
    """
    Server-side 'Judge' configuration
    """
    return WakewordConfig(
        config_name="server_judge",
        description="Production: Server Judge (ResNet18, 64 Mel, High Accuracy)",
        data=DataConfig(
            sample_rate=16000,
            audio_duration=2.0,
            n_mels=64,
            hop_length=160,
        ),
        training=TrainingConfig(
            batch_size=128,
            epochs=50,
            learning_rate=0.002,
            early_stopping_patience=10,
        ),
        model=ModelConfig(
            architecture="resnet18",
            num_classes=2,
            dropout=0.3,
        ),
        augmentation=AugmentationConfig(
            background_noise_prob=0.4,
            rir_prob=0.2,
            time_shift_prob=0.3,
        ),
        optimizer=OptimizerConfig(
            optimizer="adamw",
            weight_decay=0.01,
            mixed_precision=True,
        ),
        loss=LossConfig(
            loss_function="cross_entropy",
            label_smoothing=0.1,
            class_weights="balanced",
            hard_negative_weight=2.0,
        ),
        cmvn=CMVNConfig(enabled=True),
        streaming=StreamingConfig(
            hysteresis_high=0.6,
            hysteresis_low=0.2,
            buffer_length_ms=2000,
            smoothing_window=8,
        ),
        size_targets=SizeTargetConfig(
            max_flash_kb=0,
            max_ram_kb=0,
        ),
        calibration=CalibrationConfig(num_samples=500, positive_ratio=0.5),
    )


def get_rpi_zero2w_preset() -> WakewordConfig:
    """
    Raspberry Pi Zero 2W Satellite configuration

    Optimized for:
    - RPi Zero 2W (Cortex-A53)
    - Wyoming Satellite protocol
    - MobileNetV3 (Standard Small)
    - TFLite Float16 or Int8 deployment
    """
    return WakewordConfig(
        config_name="rpi_zero2w",
        description="Production: RPi Zero 2W Satellite (MobileNetV3, 40 Mel, Wyoming)",
        data=DataConfig(
            sample_rate=16000,
            audio_duration=1.5,
            n_mels=40,
            hop_length=160,
        ),
        training=TrainingConfig(
            batch_size=64,
            epochs=100,
            learning_rate=0.001,
            early_stopping_patience=15,
        ),
        model=ModelConfig(
            architecture="mobilenetv3",
            num_classes=2,
            dropout=0.2,
        ),
        augmentation=AugmentationConfig(
            background_noise_prob=0.5,
            rir_prob=0.3,
            time_shift_prob=0.4,
        ),
        optimizer=OptimizerConfig(
            optimizer="adamw",
            weight_decay=0.01,
            mixed_precision=True,
        ),
        loss=LossConfig(
            loss_function="cross_entropy",
            label_smoothing=0.05,
            class_weights="balanced",
            hard_negative_weight=2.5,
        ),
        qat=QATConfig(
            enabled=True,
            backend="fbgemm", # x86 for training, but convertible to arm-optimized TFLite
            start_epoch=10,
        ),
        cmvn=CMVNConfig(enabled=True),
        streaming=StreamingConfig(
            hysteresis_high=0.7,
            hysteresis_low=0.3,
            buffer_length_ms=1500,
            smoothing_window=5,
        ),
        size_targets=SizeTargetConfig(
            max_flash_kb=2048,
            max_ram_kb=1024,
        ),
        calibration=CalibrationConfig(num_samples=200, positive_ratio=0.5),
    )


def get_ultimate_accuracy_preset() -> WakewordConfig:
    """
    Ultimate Accuracy configuration for high-end x86_64 machines

    Optimized for:
    - High-end PCs/Laptops (RTX 3060+, 16GB+ RAM)
    - Maximum detection robustness
    - Large context window
    - High resolution features
    """
    return WakewordConfig(
        config_name="ultimate_accuracy",
        description="Production: x86_64 Ultimate (ResNet18, 80 Mel, 2.0s Context)",
        data=DataConfig(
            sample_rate=16000,
            audio_duration=2.0,
            n_mels=80,  # High resolution
            hop_length=160,
        ),
        training=TrainingConfig(
            batch_size=128,
            epochs=100,
            learning_rate=0.001,
            early_stopping_patience=20,
            num_workers=16, # Leverage high CPU count
        ),
        model=ModelConfig(
            architecture="resnet18",
            num_classes=2,
            dropout=0.4,
        ),
        augmentation=AugmentationConfig(
            background_noise_prob=0.6,
            rir_prob=0.5,
            time_shift_prob=0.5,
            # Aggressive SpecAugment
            n_freq_masks=3,
            n_time_masks=3,
        ),
        optimizer=OptimizerConfig(
            optimizer="adamw",
            weight_decay=0.02, # Higher regularization
            mixed_precision=True,
        ),
        loss=LossConfig(
            loss_function="focal_loss",
            focal_gamma=2.5,
            class_weights="balanced",
            hard_negative_weight=5.0, # Extremely strict on false positives
        ),
        qat=QATConfig(
            enabled=False, # Priorities FP32 accuracy for desktop/server
        ),
        cmvn=CMVNConfig(enabled=True),
        streaming=StreamingConfig(
            hysteresis_high=0.6,
            hysteresis_low=0.2,
            buffer_length_ms=2000,
            smoothing_window=10,
        ),
        size_targets=SizeTargetConfig(
            max_flash_kb=0,
            max_ram_kb=0,
        ),
        calibration=CalibrationConfig(num_samples=500, positive_ratio=0.5),
    )


def get_small_dataset_preset() -> WakewordConfig:
    """
    Utility: Small dataset configuration (<10k samples)
    """
    return WakewordConfig(
        config_name="small_dataset",
        description="Utility: Optimized for small datasets (<10k samples)",
        data=DataConfig(sample_rate=16000, audio_duration=1.5, n_mels=64),
        training=TrainingConfig(
            batch_size=16,
            epochs=100,
            learning_rate=0.0005,
            early_stopping_patience=15,
        ),
        model=ModelConfig(
            architecture="mobilenetv3",
            dropout=0.5,
        ),
        augmentation=AugmentationConfig(
            background_noise_prob=0.7,
            rir_prob=0.5,
            time_shift_prob=0.6,
        ),
        optimizer=OptimizerConfig(
            optimizer="adamw",
            weight_decay=1e-2,
        ),
        loss=LossConfig(
            loss_function="focal_loss",
            class_weights="balanced",
            hard_negative_weight=3.0,
        ),
    )


def get_fast_training_preset() -> WakewordConfig:
    """
    Utility: Fast training configuration
    """
    return WakewordConfig(
        config_name="fast_training",
        description="Utility: Fast training for quick iteration",
        data=DataConfig(
            sample_rate=16000,
            audio_duration=1.0,
            n_mels=40,
        ),
        training=TrainingConfig(
            batch_size=64,
            epochs=20,
            learning_rate=0.003,
            early_stopping_patience=5,
            use_ema=False,
        ),
        model=ModelConfig(
            architecture="mobilenetv3",
        ),
        augmentation=AugmentationConfig(
            background_noise_prob=0.3,
            rir_prob=0.1,
            time_shift_prob=0.1,
        ),
        optimizer=OptimizerConfig(
            optimizer="adamw",
            weight_decay=1e-4,
        ),
        loss=LossConfig(
            loss_function="cross_entropy",
            class_weights="balanced",
        ),
    )


# Preset registry - Simplified to 3 main target platforms
PRESETS: Dict[str, Callable[[], WakewordConfig]] = {
    # === MAIN PRODUCTION PROFILES ===
    "MCU (ESP32-S3 No-PSRAM)": get_mcu_tiny_production_preset,
    "RPI (Raspberry Pi / Wyoming Satellite)": get_rpi_zero2w_preset,
    "x86_64 (Desktop / Server)": get_ultimate_accuracy_preset,
    # === UTILITY PROFILES ===
    "Utility: Small Dataset (<10k)": get_small_dataset_preset,
    "Utility: Fast Training (Prototyping)": get_fast_training_preset,
}


def get_preset(preset_name: str) -> WakewordConfig:
    """
    Get configuration preset by name
    """
    if preset_name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Preset '{preset_name}' not found. Available presets: {available}")

    return PRESETS[preset_name]()


def list_presets() -> list:
    """
    List available preset names
    """
    return list(PRESETS.keys())