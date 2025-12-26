"""
Configuration Presets for Wake Word Training
Simplified to 3 main target platforms with industry-standard values
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


def get_esp32s3_preset() -> WakewordConfig:
    """
    ESP32-S3 Production configuration

    Target Hardware:
    - ESP32-S3 (with or without PSRAM)
    - TinyConv architecture (optimized for S3 memory)
    - Int8 Quantized deployment

    Features:
    - 64 Mel bands for high resolution
    - Knowledge Distillation from Wav2Vec2
    - Aggressive augmentation for robustness
    - QAT enabled for INT8 deployment

    Industry-aligned: focal_alpha=0.75, focal_gamma=2.0, weight_decay=0.01
    """
    return WakewordConfig(
        config_name="esp32s3",
        description="ESP32-S3 Production (TinyConv, 64 Mel, INT8 Quantized)",
        data=DataConfig(
            sample_rate=16000,
            audio_duration=1.5,
            n_mels=64,
            n_fft=512,
            hop_length=160,
            feature_type="mel",
        ),
        training=TrainingConfig(
            batch_size=64,
            epochs=200,
            learning_rate=0.001,  # Industry standard
            early_stopping_patience=25,
            use_ema=True,
            ema_decay=0.999,
        ),
        model=ModelConfig(
            architecture="tiny_conv",
            num_classes=2,
            dropout=0.2,
            tcn_num_channels=[64, 64, 64, 64],
        ),
        augmentation=AugmentationConfig(
            time_stretch_min=0.85,
            time_stretch_max=1.15,
            pitch_shift_min=-3,
            pitch_shift_max=3,
            background_noise_prob=0.8,
            noise_snr_min=3.0,  # Industry: harder conditions
            noise_snr_max=15.0,
            rir_prob=0.6,
            time_shift_prob=0.7,
            use_spec_augment=True,
            freq_mask_param=20,
            time_mask_param=40,
            n_freq_masks=2,
            n_time_masks=2,
        ),
        optimizer=OptimizerConfig(
            optimizer="adamw",
            weight_decay=0.01,  # Industry standard
            scheduler="cosine",
            warmup_epochs=5,  # Warmup for stable training
            gradient_clip=1.0,
            mixed_precision=True,
        ),
        loss=LossConfig(
            loss_function="focal_loss",
            label_smoothing=0.05,
            focal_alpha=0.75,  # Industry standard (Google RetinaNet)
            focal_gamma=2.0,  # Industry standard
            class_weights="balanced",
            hard_negative_weight=5.0,
            sampler_strategy="balanced",  # Uses (1,2,1) ratio
        ),
        qat=QATConfig(
            enabled=True,
            backend="qnnpack",
            start_epoch=15,
        ),
        distillation=DistillationConfig(
            enabled=True,
            teacher_architecture="dual",  # Wav2Vec2 + Whisper
            secondary_teacher_architecture="whisper",
            temperature=2.0,
            alpha=0.5,
        ),
        cmvn=CMVNConfig(enabled=True),
        streaming=StreamingConfig(
            hysteresis_high=0.85,
            hysteresis_low=0.35,
            buffer_length_ms=1000,
            smoothing_window=3,
            cooldown_ms=500,
        ),
        size_targets=SizeTargetConfig(
            max_flash_kb=256,
            max_ram_kb=192,
        ),
        calibration=CalibrationConfig(num_samples=300, positive_ratio=0.3),
    )


def get_rpi_zero2w_preset() -> WakewordConfig:
    """
    Raspberry Pi Zero 2W Production configuration

    Target Hardware:
    - Raspberry Pi Zero 2W (Cortex-A53)
    - Wyoming Satellite protocol compatible
    - MobileNetV3-Small architecture

    Features:
    - 40 Mel bands (industry standard for edge)
    - 1.5s context window
    - Home Assistant / Wyoming compatible
    - TFLite deployment ready

    Industry-aligned: focal_alpha=0.75, focal_gamma=2.0, weight_decay=0.01
    """
    return WakewordConfig(
        config_name="rpi_zero2w",
        description="Raspberry Pi Zero 2W (MobileNetV3, 40 Mel, Wyoming Compatible)",
        data=DataConfig(
            sample_rate=16000,
            audio_duration=1.5,
            n_mels=40,  # Industry standard for edge
            n_fft=512,
            hop_length=160,
            feature_type="mel",
        ),
        training=TrainingConfig(
            batch_size=64,
            epochs=100,
            learning_rate=0.001,  # Industry standard
            early_stopping_patience=15,
            use_ema=True,
            ema_decay=0.999,
        ),
        model=ModelConfig(
            architecture="mobilenetv3",
            num_classes=2,
            dropout=0.2,
        ),
        augmentation=AugmentationConfig(
            time_stretch_min=0.85,
            time_stretch_max=1.15,
            pitch_shift_min=-3,
            pitch_shift_max=3,
            background_noise_prob=0.6,
            noise_snr_min=3.0,  # Industry: harder conditions
            noise_snr_max=20.0,
            rir_prob=0.4,
            time_shift_prob=0.4,
            use_spec_augment=True,
            freq_mask_param=15,
            time_mask_param=30,
            n_freq_masks=2,
            n_time_masks=2,
        ),
        optimizer=OptimizerConfig(
            optimizer="adamw",
            weight_decay=0.01,  # Industry standard
            scheduler="cosine",
            warmup_epochs=5,  # Warmup for stable training
            gradient_clip=1.0,
            mixed_precision=True,
        ),
        loss=LossConfig(
            loss_function="focal_loss",
            label_smoothing=0.05,
            focal_alpha=0.75,  # Industry standard (Google RetinaNet)
            focal_gamma=2.0,  # Industry standard
            class_weights="balanced",
            hard_negative_weight=3.0,
            sampler_strategy="balanced",  # Uses (1,2,1) ratio
        ),
        qat=QATConfig(
            enabled=True,
            backend="fbgemm",  # x86 training, converts to ARM TFLite
            start_epoch=10,
        ),
        cmvn=CMVNConfig(enabled=True),
        streaming=StreamingConfig(
            hysteresis_high=0.7,
            hysteresis_low=0.3,
            buffer_length_ms=1500,
            smoothing_window=5,
            cooldown_ms=500,
        ),
        size_targets=SizeTargetConfig(
            max_flash_kb=2048,
            max_ram_kb=1024,
        ),
        calibration=CalibrationConfig(num_samples=200, positive_ratio=0.5),
    )


def get_x86_64_preset() -> WakewordConfig:
    """
    x86_64 Desktop/Server Production configuration

    Target Hardware:
    - Desktop PCs / Laptops (RTX 3060+, 16GB+ RAM)
    - Servers / Home Assistant hosts
    - Maximum accuracy priority

    Features:
    - 80 Mel bands for highest resolution
    - 2.0s context window
    - ResNet18 architecture
    - FP32 inference (no quantization)
    - Aggressive SpecAugment

    Industry-aligned: focal_alpha=0.75, focal_gamma=2.0, weight_decay=0.02
    """
    return WakewordConfig(
        config_name="x86_64",
        description="x86_64 Desktop/Server (ResNet18, 80 Mel, Maximum Accuracy)",
        data=DataConfig(
            sample_rate=16000,
            audio_duration=2.0,
            n_mels=80,  # High resolution
            n_fft=512,
            hop_length=160,
            feature_type="mel",
        ),
        training=TrainingConfig(
            batch_size=128,
            epochs=100,
            learning_rate=0.001,  # Industry standard
            early_stopping_patience=20,
            num_workers=16,  # Leverage high CPU count
            use_ema=True,
            ema_decay=0.999,
        ),
        model=ModelConfig(
            architecture="resnet18",
            num_classes=2,
            dropout=0.4,
        ),
        augmentation=AugmentationConfig(
            time_stretch_min=0.85,
            time_stretch_max=1.15,
            pitch_shift_min=-3,
            pitch_shift_max=3,
            background_noise_prob=0.6,
            noise_snr_min=3.0,  # Industry: harder conditions
            noise_snr_max=20.0,
            rir_prob=0.5,
            time_shift_prob=0.5,
            use_spec_augment=True,
            freq_mask_param=20,
            time_mask_param=40,
            # Aggressive SpecAugment
            n_freq_masks=3,
            n_time_masks=3,
        ),
        optimizer=OptimizerConfig(
            optimizer="adamw",
            weight_decay=0.02,  # Higher regularization for large models
            scheduler="cosine",
            warmup_epochs=5,  # ✅ Correct location for warmup
            gradient_clip=1.0,
            mixed_precision=True,
        ),
        loss=LossConfig(
            loss_function="focal_loss",
            label_smoothing=0.1,
            focal_alpha=0.75,  # Industry standard (Google RetinaNet)
            focal_gamma=2.0,  # Industry standard
            class_weights="balanced",
            hard_negative_weight=5.0,
            sampler_strategy="balanced",  # Uses (1,2,1) ratio
        ),
        qat=QATConfig(
            enabled=False,  # FP32 for maximum accuracy
        ),
        cmvn=CMVNConfig(enabled=True),
        streaming=StreamingConfig(
            hysteresis_high=0.6,
            hysteresis_low=0.2,
            buffer_length_ms=2000,
            smoothing_window=10,
            cooldown_ms=500,
        ),
        size_targets=SizeTargetConfig(
            max_flash_kb=0,  # No limit
            max_ram_kb=0,
        ),
        calibration=CalibrationConfig(num_samples=500, positive_ratio=0.5),
    )


# ============================================================================
# PRESET REGISTRY - 3 Target Platforms Only
# ============================================================================
PRESETS: Dict[str, Callable[[], WakewordConfig]] = {
    "ESP32-S3": get_esp32s3_preset,
    "Raspberry Pi Zero 2W": get_rpi_zero2w_preset,
    "x86_64 (Desktop/Server)": get_x86_64_preset,
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
