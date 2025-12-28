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
# STRATEGY PRESETS - For Specialized Training Workflows
# ============================================================================


def get_hard_negative_refinement_preset() -> WakewordConfig:
    """
    Strategy A: Hard Negative Refinement configuration

    Purpose:
    - Fine-tune a model on mined hard negatives (false positives)
    - Reduce false alarm rate without losing sensitivity
    - Quick refinement training cycle

    Key Features:
    - High hard_negative_weight (10.0) to heavily penalize false positives
    - Lower learning rate for stable fine-tuning
    - Shorter training epochs (30) for quick iteration
    - Aggressive focal loss for imbalanced hard negative data
    - No QAT (refinement phase, quantize after)
    - Conservative augmentation (don't want to augment away the hard patterns)

    Usage:
    This preset is auto-generated when mining hard negatives from the
    evaluation panel. It creates a training profile specifically designed
    to teach the model about confusing non-wakeword sounds.
    """
    return WakewordConfig(
        config_name="strategy_a_hard_negative",
        description="Strategy A: Hard Negative Refinement (Reduce False Positives)",
        data=DataConfig(
            sample_rate=16000,
            audio_duration=1.5,
            n_mels=64,  # Match ESP32-S3 default for compatibility
            n_fft=512,
            hop_length=160,
            feature_type="mel",
        ),
        training=TrainingConfig(
            batch_size=32,  # Smaller batch for fine-tuning
            epochs=30,  # Short refinement cycle
            learning_rate=0.0001,  # Lower LR for stable fine-tuning
            early_stopping_patience=10,  # Quick early stop
            use_ema=True,
            ema_decay=0.999,
        ),
        model=ModelConfig(
            architecture="tiny_conv",  # Inherit from loaded model in practice
            num_classes=2,
            dropout=0.3,  # Slightly higher dropout for regularization
            tcn_num_channels=[64, 64, 64, 64],
        ),
        augmentation=AugmentationConfig(
            # Conservative augmentation - we want to learn the exact patterns
            time_stretch_min=0.95,  # Less aggressive stretch
            time_stretch_max=1.05,
            pitch_shift_min=-1,  # Less aggressive pitch
            pitch_shift_max=1,
            background_noise_prob=0.3,  # Lower noise - learn clean patterns first
            noise_snr_min=10.0,  # Higher SNR (easier)
            noise_snr_max=25.0,
            rir_prob=0.2,  # Less reverb
            time_shift_prob=0.3,
            use_spec_augment=True,  # Keep SpecAugment for regularization
            freq_mask_param=10,  # Smaller masks
            time_mask_param=20,
            n_freq_masks=1,
            n_time_masks=1,
        ),
        optimizer=OptimizerConfig(
            optimizer="adamw",
            weight_decay=0.02,  # Higher regularization for fine-tuning
            scheduler="cosine",
            warmup_epochs=2,  # Short warmup
            gradient_clip=0.5,  # Tighter gradient clipping
            mixed_precision=True,
        ),
        loss=LossConfig(
            loss_function="focal_loss",
            label_smoothing=0.1,  # Higher smoothing for robustness
            focal_alpha=0.85,  # Higher alpha = more focus on minority (wakeword)
            focal_gamma=3.0,  # Higher gamma = more focus on hard examples
            class_weights="balanced",
            hard_negative_weight=10.0,  # KEY: Heavy penalty on hard negatives
            sampler_strategy="hard_negative_focused",  # Prioritize hard negatives
        ),
        qat=QATConfig(
            enabled=False,  # No QAT during refinement - do it after
        ),
        distillation=DistillationConfig(
            enabled=False,  # No distillation during refinement
        ),
        cmvn=CMVNConfig(enabled=True),
        streaming=StreamingConfig(
            hysteresis_high=0.9,  # Very high threshold after refinement
            hysteresis_low=0.4,
            buffer_length_ms=1000,
            smoothing_window=5,
            cooldown_ms=1000,  # Longer cooldown to prevent rapid triggers
        ),
        size_targets=SizeTargetConfig(
            max_flash_kb=0,  # No limit during refinement
            max_ram_kb=0,
        ),
        calibration=CalibrationConfig(num_samples=500, positive_ratio=0.3),
    )


def get_recall_enhancement_preset() -> WakewordConfig:
    """
    Strategy B: Recall Enhancement (Fix High FNR)

    Problem Symptoms:
    - Missing real wakeword detections
    - FNR > 5%
    - Recall < 90%
    - Works on clear speech, fails on variations (whispers, accents, distance)

    Root Cause:
    Model is too strict / hasn't learned enough positive variations

    Key Fixes:
    - Lower focal_alpha (0.5) = balanced class focus, not over-penalizing positives
    - Lower focal_gamma (1.5) = less aggressive hard mining
    - Aggressive augmentation on time/pitch to learn variations
    - Train on noisier positive samples
    """
    return WakewordConfig(
        config_name="strategy_b_recall_enhancement",
        description="Strategy B: Recall Enhancement (Fix High FNR / Low Recall)",
        data=DataConfig(
            sample_rate=16000,
            audio_duration=1.5,
            n_mels=64,
            n_fft=512,
            hop_length=160,
            feature_type="mel",
        ),
        training=TrainingConfig(
            batch_size=32,
            epochs=40,  # Slightly longer to learn variations
            learning_rate=0.0001,  # Gentle fine-tuning
            early_stopping_patience=12,
            use_ema=True,
            ema_decay=0.999,
        ),
        model=ModelConfig(
            architecture="tiny_conv",
            num_classes=2,
            dropout=0.2,  # Normal dropout - don't over-regularize
            tcn_num_channels=[64, 64, 64, 64],
        ),
        augmentation=AugmentationConfig(
            # Aggressive augmentation to learn positive variations
            time_stretch_min=0.75,  # Wide speed range
            time_stretch_max=1.25,
            pitch_shift_min=-5,  # Wide pitch range (accents, speakers)
            pitch_shift_max=5,
            background_noise_prob=0.6,  # Train positives with noise
            noise_snr_min=5.0,  # Noisier conditions
            noise_snr_max=20.0,
            rir_prob=0.5,  # Reverb for distance variation
            time_shift_prob=0.5,
            use_spec_augment=True,
            freq_mask_param=15,
            time_mask_param=30,
            n_freq_masks=2,
            n_time_masks=2,
        ),
        optimizer=OptimizerConfig(
            optimizer="adamw",
            weight_decay=0.01,  # Normal regularization
            scheduler="cosine",
            warmup_epochs=3,
            gradient_clip=1.0,
            mixed_precision=True,
        ),
        loss=LossConfig(
            loss_function="focal_loss",
            label_smoothing=0.15,  # Higher smoothing - softer targets
            focal_alpha=0.5,  # KEY: Balanced focus (not biased to negatives)
            focal_gamma=1.5,  # KEY: Less aggressive hard mining
            class_weights="balanced",
            hard_negative_weight=1.0,  # Normal weight - don't over-penalize FPs
            sampler_strategy="balanced",
        ),
        qat=QATConfig(enabled=False),
        distillation=DistillationConfig(enabled=False),
        cmvn=CMVNConfig(enabled=True),
        streaming=StreamingConfig(
            hysteresis_high=0.6,  # Lower threshold for better recall
            hysteresis_low=0.25,
            buffer_length_ms=1000,
            smoothing_window=3,
            cooldown_ms=500,
        ),
        size_targets=SizeTargetConfig(max_flash_kb=0, max_ram_kb=0),
        calibration=CalibrationConfig(num_samples=500, positive_ratio=0.5),
    )


def get_anti_overfitting_preset() -> WakewordConfig:
    """
    Strategy C: Anti-Overfitting Refinement

    Problem Symptoms:
    - Training accuracy: 99%+
    - Validation accuracy: 85-90% (big gap > 10%)
    - Train loss keeps decreasing, Val loss starts increasing
    - Great on training data, poor on new/unseen data

    Root Cause:
    Model memorized training samples instead of learning generalizable patterns

    Key Fixes:
    - High dropout (0.5) = heavy regularization
    - High weight_decay (0.05) = strong L2 penalty
    - Very low learning rate = don't reinforce memorized patterns
    - Aggressive augmentation = force generalization
    - Short epochs = stop before overfitting more
    """
    return WakewordConfig(
        config_name="strategy_c_anti_overfitting",
        description="Strategy C: Anti-Overfitting (Fix Train >> Val Gap)",
        data=DataConfig(
            sample_rate=16000,
            audio_duration=1.5,
            n_mels=64,
            n_fft=512,
            hop_length=160,
            feature_type="mel",
        ),
        training=TrainingConfig(
            batch_size=64,  # Larger batch for stability
            epochs=20,  # SHORT - overfit more = worse
            learning_rate=0.00005,  # Very gentle updates
            early_stopping_patience=5,  # Aggressive early stop
            use_ema=True,
            ema_decay=0.9995,  # Stronger smoothing
        ),
        model=ModelConfig(
            architecture="tiny_conv",
            num_classes=2,
            dropout=0.5,  # KEY: Heavy dropout
            tcn_num_channels=[64, 64, 64, 64],
        ),
        augmentation=AugmentationConfig(
            # Maximum augmentation to force generalization
            time_stretch_min=0.80,
            time_stretch_max=1.20,
            pitch_shift_min=-4,
            pitch_shift_max=4,
            background_noise_prob=0.8,
            noise_snr_min=3.0,
            noise_snr_max=20.0,
            rir_prob=0.6,
            time_shift_prob=0.6,
            use_spec_augment=True,
            freq_mask_param=25,  # Larger masks
            time_mask_param=50,
            n_freq_masks=3,  # More masks
            n_time_masks=3,
        ),
        optimizer=OptimizerConfig(
            optimizer="adamw",
            weight_decay=0.05,  # KEY: Strong L2 regularization
            scheduler="cosine",
            warmup_epochs=1,
            gradient_clip=0.3,  # Tight gradient clipping
            mixed_precision=True,
        ),
        loss=LossConfig(
            loss_function="focal_loss",
            label_smoothing=0.15,  # Prevent overconfidence
            focal_alpha=0.75,
            focal_gamma=2.0,
            class_weights="balanced",
            hard_negative_weight=3.0,
            sampler_strategy="balanced",
        ),
        qat=QATConfig(enabled=False),
        distillation=DistillationConfig(enabled=False),
        cmvn=CMVNConfig(enabled=True),
        streaming=StreamingConfig(
            hysteresis_high=0.7,
            hysteresis_low=0.3,
            buffer_length_ms=1000,
            smoothing_window=5,
            cooldown_ms=500,
        ),
        size_targets=SizeTargetConfig(max_flash_kb=0, max_ram_kb=0),
        calibration=CalibrationConfig(num_samples=300, positive_ratio=0.4),
    )


def get_noise_robustness_preset() -> WakewordConfig:
    """
    Strategy D: Noise Robustness Enhancement

    Problem Symptoms:
    - Good accuracy in quiet test environments
    - Performance degrades significantly with background noise
    - Real-world deployment much worse than test metrics
    - FPR/FNR spike in noisy conditions (kitchen, TV, traffic)

    Root Cause:
    Model trained on too-clean audio, doesn't generalize to real-world noise

    Key Fixes:
    - Very high noise probability (0.9)
    - Very low SNR (0-15 dB) = train on heavily noisy audio
    - High reverb probability = room acoustics
    - Longer training to learn noise patterns
    """
    return WakewordConfig(
        config_name="strategy_d_noise_robustness",
        description="Strategy D: Noise Robustness (Fix Noisy Environment Performance)",
        data=DataConfig(
            sample_rate=16000,
            audio_duration=1.5,
            n_mels=64,
            n_fft=512,
            hop_length=160,
            feature_type="mel",
        ),
        training=TrainingConfig(
            batch_size=32,
            epochs=50,  # Longer to learn noise patterns
            learning_rate=0.0001,
            early_stopping_patience=15,
            use_ema=True,
            ema_decay=0.999,
        ),
        model=ModelConfig(
            architecture="tiny_conv",
            num_classes=2,
            dropout=0.3,
            tcn_num_channels=[64, 64, 64, 64],
        ),
        augmentation=AugmentationConfig(
            time_stretch_min=0.90,
            time_stretch_max=1.10,
            pitch_shift_min=-2,
            pitch_shift_max=2,
            # KEY: Extreme noise augmentation
            background_noise_prob=0.9,  # Almost always add noise
            noise_snr_min=0.0,  # KEY: Very noisy (0 dB = equal noise!)
            noise_snr_max=15.0,  # Range up to moderate
            rir_prob=0.7,  # Frequent reverb
            rir_dry_wet_max=0.8,  # Strong reverb when applied
            time_shift_prob=0.4,
            use_spec_augment=True,
            freq_mask_param=20,
            time_mask_param=40,
            n_freq_masks=2,
            n_time_masks=2,
        ),
        optimizer=OptimizerConfig(
            optimizer="adamw",
            weight_decay=0.02,
            scheduler="cosine",
            warmup_epochs=5,
            gradient_clip=1.0,
            mixed_precision=True,
        ),
        loss=LossConfig(
            loss_function="focal_loss",
            label_smoothing=0.1,
            focal_alpha=0.75,
            focal_gamma=2.0,
            class_weights="balanced",
            hard_negative_weight=5.0,  # Penalize noisy FPs
            sampler_strategy="balanced",
        ),
        qat=QATConfig(enabled=False),
        distillation=DistillationConfig(enabled=False),
        cmvn=CMVNConfig(enabled=True),
        streaming=StreamingConfig(
            hysteresis_high=0.75,
            hysteresis_low=0.35,
            buffer_length_ms=1200,
            smoothing_window=5,
            cooldown_ms=700,
        ),
        size_targets=SizeTargetConfig(max_flash_kb=0, max_ram_kb=0),
        calibration=CalibrationConfig(num_samples=500, positive_ratio=0.4),
    )


def get_f1_balance_preset() -> WakewordConfig:
    """
    Strategy E: F1 Balance Optimization

    Problem Symptoms:
    - F1 Score < 85%
    - Neither precision nor recall are satisfactory
    - Threshold selection is difficult (all thresholds seem bad)
    - ROC-AUC is moderate (0.90-0.95) - model has potential but isn't balanced

    Root Cause:
    Overall suboptimal training - needs balanced refinement across all aspects

    Key Fixes:
    - Balanced focal parameters (alpha=0.75, gamma=2.0)
    - Balanced sampling strategy
    - Moderate augmentation across all types
    - EMA for smooth model averaging
    - Longer training with patience
    """
    return WakewordConfig(
        config_name="strategy_e_f1_balance",
        description="Strategy E: F1 Balance Optimization (Fix Poor Overall F1)",
        data=DataConfig(
            sample_rate=16000,
            audio_duration=1.5,
            n_mels=64,
            n_fft=512,
            hop_length=160,
            feature_type="mel",
        ),
        training=TrainingConfig(
            batch_size=48,
            epochs=60,  # Longer refinement
            learning_rate=0.0002,  # Moderate LR
            early_stopping_patience=15,
            use_ema=True,
            ema_decay=0.999,
        ),
        model=ModelConfig(
            architecture="tiny_conv",
            num_classes=2,
            dropout=0.25,  # Balanced dropout
            tcn_num_channels=[64, 64, 64, 64],
        ),
        augmentation=AugmentationConfig(
            # Balanced augmentation across all types
            time_stretch_min=0.85,
            time_stretch_max=1.15,
            pitch_shift_min=-3,
            pitch_shift_max=3,
            background_noise_prob=0.6,
            noise_snr_min=5.0,
            noise_snr_max=20.0,
            rir_prob=0.4,
            time_shift_prob=0.5,
            use_spec_augment=True,
            freq_mask_param=15,
            time_mask_param=30,
            n_freq_masks=2,
            n_time_masks=2,
        ),
        optimizer=OptimizerConfig(
            optimizer="adamw",
            weight_decay=0.015,  # Balanced regularization
            scheduler="cosine",
            warmup_epochs=5,
            gradient_clip=1.0,
            mixed_precision=True,
        ),
        loss=LossConfig(
            loss_function="focal_loss",
            label_smoothing=0.1,
            focal_alpha=0.75,  # Industry standard balanced
            focal_gamma=2.0,  # Industry standard
            class_weights="balanced",  # Auto-balance classes
            hard_negative_weight=3.0,  # Moderate
            sampler_strategy="balanced",  # Equal sampling
        ),
        qat=QATConfig(enabled=False),
        distillation=DistillationConfig(enabled=False),
        cmvn=CMVNConfig(enabled=True),
        streaming=StreamingConfig(
            hysteresis_high=0.7,
            hysteresis_low=0.3,
            buffer_length_ms=1000,
            smoothing_window=5,
            cooldown_ms=500,
        ),
        size_targets=SizeTargetConfig(max_flash_kb=0, max_ram_kb=0),
        calibration=CalibrationConfig(num_samples=500, positive_ratio=0.5),
    )


# ============================================================================
# PRESET REGISTRY - Hardware Platforms + Training Strategies
# ============================================================================
PRESETS: Dict[str, Callable[[], WakewordConfig]] = {
    # Hardware Target Presets
    "ESP32-S3": get_esp32s3_preset,
    "Raspberry Pi Zero 2W": get_rpi_zero2w_preset,
    "x86_64 (Desktop/Server)": get_x86_64_preset,
    # Refinement Strategy Presets (Resume Training to Fix Problems)
    "Strategy A: Hard Negative Refinement": get_hard_negative_refinement_preset,
    "Strategy B: Recall Enhancement": get_recall_enhancement_preset,
    "Strategy C: Anti-Overfitting": get_anti_overfitting_preset,
    "Strategy D: Noise Robustness": get_noise_robustness_preset,
    "Strategy E: F1 Balance": get_f1_balance_preset,
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
