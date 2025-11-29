"""
Panel 2: Configuration Management
- Basic and advanced parameter editing
- Configuration presets
- Save/load configuration
"""
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import gradio as gr

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.defaults import WakewordConfig, QATConfig, DistillationConfig
from src.config.logger import get_data_logger
from src.config.presets import get_preset, list_presets
from src.config.validator import ConfigValidator
from src.exceptions import WakewordException

logger = get_data_logger()

# Global state for current configuration
_current_config = None


def create_config_panel(state: gr.State = None) -> gr.Blocks:
    """
    Create Panel 2: Configuration Management

    Args:
        state: Global state dictionary for sharing config between panels

    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks() as panel:
        gr.Markdown("# ⚙️ Training Configuration")
        gr.Markdown("Configure all training parameters.")

        with gr.Row():
            preset_dropdown = gr.Dropdown(
                choices=list_presets(),
                value="Default",
                label="Configuration Preset",
                info="Select a preset optimized for your use case",
            )
            load_preset_btn = gr.Button("📥 Load Preset", variant="secondary")

        with gr.Row():
            save_config_btn = gr.Button("💾 Save Configuration", variant="primary")
            load_config_btn = gr.Button("📂 Load Configuration", variant="secondary")
            reset_config_btn = gr.Button("🔄 Reset to Defaults", variant="secondary")
            validate_btn = gr.Button("✅ Validate Configuration", variant="secondary")

        with gr.Tabs():
            # =================================================================
            # TAB 1: DATA PIPELINE
            # =================================================================
            with gr.TabItem("Data Pipeline"):
                with gr.Group():
                    gr.Markdown("### 🎵 Audio Settings")
                    with gr.Row():
                        sample_rate = gr.Number(
                            label="Sample Rate (Hz)",
                            value=16000,
                            info="Audio sample rate (16kHz recommended)",
                        )
                        audio_duration = gr.Number(
                            label="Audio Duration (s)",
                            value=1.5,
                            step=0.1,
                            info="Length of audio clips",
                        )
                        n_mfcc = gr.Slider(
                            minimum=0, maximum=80, value=0, step=1,
                            label="MFCC Coefficients",
                            info="0 to use mel only",
                        )
                    with gr.Row():
                        n_fft = gr.Number(label="FFT Size", value=400)
                        hop_length = gr.Number(label="Hop Length", value=160)
                        n_mels = gr.Slider(
                            minimum=32, maximum=256, value=64, step=1,
                            label="Mel Bands",
                        )

                with gr.Group():
                    gr.Markdown("### 💾 Feature Loading (NPY)")
                    gr.Markdown("Performance: 40-60% faster training for large datasets")
                    with gr.Row():
                        use_precomputed_features_for_training = gr.Checkbox(
                            label="Use Precomputed NPY", value=True,
                            info="Load features from .npy files"
                        )
                        npy_cache_features = gr.Checkbox(
                            label="Cache in RAM", value=True,
                            info="Faster but uses more memory"
                        )
                        fallback_to_audio = gr.Checkbox(
                            label="Fallback to Audio", value=False,
                            info="Load raw audio if .npy missing"
                        )
                    with gr.Row():
                        npy_feature_dir = gr.Textbox(
                            label="NPY Directory", value="data/npy"
                        )
                        npy_feature_type = gr.Dropdown(
                            choices=["mel", "mfcc"], value="mel",
                            label="Feature Type"
                        )

                with gr.Group():
                    gr.Markdown("### 🛠️ Augmentation")
                    
                    gr.Markdown("**Time Domain**")
                    with gr.Row():
                        time_stretch_min = gr.Number(
                            label="Time Stretch Min", value=0.9, step=0.01
                        )
                        time_stretch_max = gr.Number(
                            label="Time Stretch Max", value=1.1, step=0.01
                        )
                        pitch_shift_min = gr.Number(
                            label="Pitch Shift Min", value=-2, step=1
                        )
                        pitch_shift_max = gr.Number(
                            label="Pitch Shift Max", value=2, step=1
                        )
                    
                    with gr.Row():
                        time_shift_prob = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.5, step=0.1,
                            label="Time Shift Prob"
                        )
                        time_shift_min_ms = gr.Number(
                            label="Shift Min (ms)", value=-100, step=10
                        )
                        time_shift_max_ms = gr.Number(
                            label="Shift Max (ms)", value=100, step=10
                        )

                    gr.Markdown("**Noise & RIR**")
                    with gr.Row():
                        background_noise_prob = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                            label="Bg Noise Prob"
                        )
                        noise_snr_min = gr.Number(label="SNR Min (dB)", value=5, step=1)
                        noise_snr_max = gr.Number(label="SNR Max (dB)", value=20, step=1)
                    
                    with gr.Row():
                        rir_prob = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.25, step=0.05,
                            label="RIR Prob"
                        )
                        rir_dry_wet_min = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.3, step=0.05,
                            label="Dry/Wet Min"
                        )
                        rir_dry_wet_max = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.7, step=0.05,
                            label="Dry/Wet Max"
                        )
                        rir_dry_wet_strategy = gr.Dropdown(
                            choices=["random", "fixed", "adaptive"], value="random",
                            label="Mix Strategy"
                        )

                    gr.Markdown("**SpecAugment**")
                    with gr.Row():
                        use_spec_augment = gr.Checkbox(label="Enable SpecAugment", value=True)
                        freq_mask_param = gr.Number(label="Freq Mask", value=15)
                        time_mask_param = gr.Number(label="Time Mask", value=30)
                        n_freq_masks = gr.Number(label="N Freq Masks", value=2)
                        n_time_masks = gr.Number(label="N Time Masks", value=2)

            # =================================================================
            # TAB 2: MODEL & TRAINING
            # =================================================================
            with gr.TabItem("Model & Training"):
                with gr.Group():
                    gr.Markdown("### 🧠 Model Architecture")
                    with gr.Row():
                        architecture = gr.Dropdown(
                            choices=["resnet18", "mobilenetv3", "lstm", "gru", "tcn", "tiny_conv", "cd_dnn"],
                            value="resnet18",
                            label="Architecture",
                        )
                        num_classes = gr.Number(label="Classes", value=2)
                        dropout = gr.Slider(
                            minimum=0.0, maximum=0.9, value=0.3, step=0.05,
                            label="Dropout"
                        )

                    gr.Markdown("**Architecture Specifics**")
                    with gr.Row():
                        # RNN
                        hidden_size = gr.Number(label="RNN Hidden Size", value=128)
                        num_layers = gr.Slider(minimum=1, maximum=5, value=2, step=1, label="RNN Layers")
                        bidirectional = gr.Checkbox(label="RNN Bidirectional", value=True)
                    
                    with gr.Row():
                        # TCN
                        tcn_num_channels = gr.Textbox(label="TCN Channels", value="64, 128, 256")
                        tcn_kernel_size = gr.Number(label="TCN Kernel", value=3)
                        tcn_dropout = gr.Slider(minimum=0.0, maximum=0.9, value=0.3, step=0.05, label="TCN Dropout")

                    with gr.Row():
                        # CD-DNN
                        cddnn_hidden_layers = gr.Textbox(label="CD-DNN Layers", value="512, 256, 128")
                        cddnn_context_frames = gr.Number(label="Context Frames", value=50)
                        cddnn_dropout = gr.Slider(minimum=0.0, maximum=0.9, value=0.3, step=0.05, label="CD-DNN Dropout")

                with gr.Group():
                    gr.Markdown("### 🏋️ Training Loop")
                    with gr.Row():
                        batch_size = gr.Slider(
                            minimum=8, maximum=1024, value=64, step=8,
                            label="Batch Size"
                        )
                        epochs = gr.Slider(
                            minimum=10, maximum=200, value=80, step=5,
                            label="Epochs"
                        )
                        num_workers = gr.Slider(
                            minimum=0, maximum=32, value=16, step=1,
                            label="Workers"
                        )
                        mixed_precision = gr.Checkbox(
                            label="Mixed Precision (FP16)", value=True
                        )

                with gr.Group():
                    gr.Markdown("### 🚀 Advanced Training")
                    with gr.Row():
                        # EMA
                        use_ema = gr.Checkbox(label="Use EMA", value=True)
                        ema_decay = gr.Number(label="EMA Decay", value=0.999, step=0.0001)
                        ema_final_decay = gr.Number(label="EMA Final Decay", value=0.9995, step=0.0001)
                        ema_final_epochs = gr.Number(label="EMA Final Epochs", value=10)
                    
                    with gr.Row():
                        checkpoint_frequency = gr.Dropdown(
                            choices=["every_epoch", "every_5_epochs", "every_10_epochs", "best_only"],
                            value="best_only",
                            label="Checkpoint Freq"
                        )
                        early_stopping = gr.Slider(
                            minimum=5, maximum=50, value=15, step=1,
                            label="Early Stopping"
                        )
                        metric_window_size = gr.Number(label="Metric Window", value=100)

            # =================================================================
            # TAB 3: OPTIMIZATION & LOSS
            # =================================================================
            with gr.TabItem("Optimization & Loss"):
                with gr.Group():
                    gr.Markdown("### 📉 Optimizer & Scheduler")
                    with gr.Row():
                        optimizer = gr.Dropdown(
                            choices=["adam", "sgd", "adamw"], value="adamw",
                            label="Optimizer"
                        )
                        learning_rate = gr.Number(
                            label="Learning Rate", value=0.0003, step=0.00001,
                            precision=5
                        )
                        weight_decay = gr.Number(
                            label="Weight Decay", value=1e-4, step=1e-5,
                            precision=5
                        )
                        gradient_clip = gr.Number(label="Grad Clip", value=1.0, step=0.1)

                    with gr.Row():
                        momentum = gr.Number(label="Momentum", value=0.9, step=0.01)
                        warmup_epochs = gr.Number(label="Warmup Epochs", value=3)
                        min_lr = gr.Number(label="Min LR", value=1e-6, step=1e-7, precision=7)

                    with gr.Row():
                        scheduler = gr.Dropdown(
                            choices=["cosine", "step", "plateau", "none"], value="cosine",
                            label="Scheduler"
                        )
                        step_size = gr.Number(label="Step Size", value=10)
                        scheduler_gamma = gr.Number(label="Gamma", value=0.5, step=0.1)
                        scheduler_patience = gr.Number(label="Patience", value=5)
                        scheduler_factor = gr.Number(label="Factor", value=0.5, step=0.1)

                with gr.Group():
                    gr.Markdown("### 🎯 Loss Function")
                    with gr.Row():
                        loss_function = gr.Dropdown(
                            choices=["cross_entropy", "focal_loss", "triplet_loss"],
                            value="cross_entropy",
                            label="Loss Function"
                        )
                        label_smoothing = gr.Slider(
                            minimum=0.0, maximum=0.3, value=0.05, step=0.01,
                            label="Label Smoothing"
                        )
                        class_weights = gr.Dropdown(
                            choices=["balanced", "none", "custom"], value="balanced",
                            label="Class Weights"
                        )
                        hard_negative_weight = gr.Number(
                            label="Hard Neg Weight", value=1.5, step=0.1
                        )

                    with gr.Row():
                        focal_alpha = gr.Number(label="Focal Alpha", value=0.25, step=0.01)
                        focal_gamma = gr.Number(label="Focal Gamma", value=2.0, step=0.1)
                        triplet_margin = gr.Slider(
                            minimum=0.1, maximum=5.0, value=1.0, step=0.1,
                            label="Triplet Margin"
                        )
                        sampler_strategy = gr.Dropdown(
                            choices=["weighted", "balanced", "none"], value="weighted",
                            label="Sampler"
                        )
                    
                    with gr.Row():
                        class_weight_min = gr.Number(label="Class Weight Min", value=0.1, step=0.1)
                        class_weight_max = gr.Number(label="Class Weight Max", value=100.0, step=10.0)

                with gr.Group():
                    gr.Markdown("### 🧪 Advanced Optimization")
                    with gr.Row():
                        # QAT
                        qat_enabled = gr.Checkbox(label="Enable QAT", value=False)
                        qat_backend = gr.Dropdown(
                            choices=["fbgemm", "qnnpack"], value="fbgemm",
                            label="QAT Backend"
                        )
                        qat_start_epoch = gr.Number(label="QAT Start Epoch", value=5)

                    with gr.Row():
                        # Distillation
                        distillation_enabled = gr.Checkbox(label="Enable Distillation", value=False)
                        teacher_arch = gr.Dropdown(
                            choices=["wav2vec2"], value="wav2vec2",
                            label="Teacher Arch"
                        )
                        dist_temp = gr.Slider(
                            minimum=1.0, maximum=10.0, value=2.0, step=0.5,
                            label="Distillation Temp"
                        )
                        dist_alpha = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.5, step=0.1,
                            label="Distillation Alpha"
                        )

        gr.Markdown("---")

        with gr.Row():
            config_status = gr.Textbox(
                label="Configuration Status",
                value="Configuration ready. Modify parameters above or load a preset.",
                lines=3,
                interactive=False,
            )

        with gr.Row():
            validation_report = gr.Textbox(
                label="Validation Report",
                value="Click 'Validate Configuration' to check parameters.",
                lines=10,
                interactive=False,
                visible=False,
            )

        # Collect all inputs for easier handling
        all_inputs = [
            # Data (0-5)
            sample_rate, audio_duration, n_mfcc, n_fft, hop_length, n_mels,
            # NPY (6-10)
            use_precomputed_features_for_training, npy_cache_features, fallback_to_audio, npy_feature_dir, npy_feature_type,
            # Training (11-14)
            batch_size, epochs, learning_rate, early_stopping,
            # Model (15-20)
            architecture, num_classes, dropout, hidden_size, num_layers, bidirectional,
            # Augmentation (21-31)
            time_stretch_min, time_stretch_max, pitch_shift_min, pitch_shift_max, background_noise_prob, rir_prob, noise_snr_min, noise_snr_max,
            rir_dry_wet_min, rir_dry_wet_max, rir_dry_wet_strategy,
            # SpecAugment (32-36)
            use_spec_augment, freq_mask_param, time_mask_param, n_freq_masks, n_time_masks,
            # Optimizer (37-48)
            optimizer, scheduler, weight_decay, gradient_clip, mixed_precision,
            momentum, warmup_epochs, min_lr, step_size, scheduler_gamma, scheduler_patience, scheduler_factor,
            # Workers (49)
            num_workers,
            # Loss (50-56)
            loss_function, label_smoothing, class_weights, hard_negative_weight,
            focal_alpha, focal_gamma, sampler_strategy,
            # Checkpoint (57)
            checkpoint_frequency,
            # Time Shift (58-60)
            time_shift_prob, time_shift_min_ms, time_shift_max_ms,
            # Triplet (61)
            triplet_margin,
            # QAT (62-64)
            qat_enabled, qat_backend, qat_start_epoch,
            # Distillation (65-68)
            distillation_enabled, teacher_arch, dist_temp, dist_alpha,
            # New Params (69-82)
            use_ema, ema_decay, ema_final_decay, ema_final_epochs, metric_window_size,
            tcn_num_channels, tcn_kernel_size, tcn_dropout,
            cddnn_hidden_layers, cddnn_context_frames, cddnn_dropout,
            class_weight_min, class_weight_max
        ]

        # Event handlers with full implementation
        def _params_to_config(params):
            """Convert UI parameters to WakewordConfig"""
            from src.config.defaults import (
                AugmentationConfig,
                DataConfig,
                LossConfig,
                ModelConfig,
                OptimizerConfig,
                TrainingConfig,
                QATConfig,
                DistillationConfig,
            )

            return WakewordConfig(
                config_name="custom",
                description="Custom configuration from UI",
                data=DataConfig(
                    sample_rate=int(params[0]),
                    audio_duration=float(params[1]),
                    n_mfcc=int(params[2]),
                    n_fft=int(params[3]),
                    hop_length=int(params[4]),
                    n_mels=int(params[5]),
                    use_precomputed_features_for_training=bool(params[6]),
                    npy_cache_features=bool(params[7]),
                    fallback_to_audio=bool(params[8]),
                    npy_feature_dir=str(params[9]),
                    npy_feature_type=str(params[10]),
                ),
                training=TrainingConfig(
                    batch_size=int(params[11]),
                    epochs=int(params[12]),
                    learning_rate=float(params[13]),
                    early_stopping_patience=int(params[14]),
                    num_workers=int(params[49]),
                    checkpoint_frequency=params[57],
                    use_ema=bool(params[69]),
                    ema_decay=float(params[70]),
                    ema_final_decay=float(params[71]),
                    ema_final_epochs=int(params[72]),
                    metric_window_size=int(params[73]),
                ),
                model=ModelConfig(
                    architecture=params[15],
                    num_classes=int(params[16]),
                    dropout=float(params[17]),
                    hidden_size=int(params[18]),
                    num_layers=int(params[19]),
                    bidirectional=bool(params[20]),
                    tcn_num_channels=[int(x.strip()) for x in params[74].split(",") if x.strip()],
                    tcn_kernel_size=int(params[75]),
                    tcn_dropout=float(params[76]),
                    cddnn_hidden_layers=[int(x.strip()) for x in params[77].split(",") if x.strip()],
                    cddnn_context_frames=int(params[78]),
                    cddnn_dropout=float(params[79]),
                ),
                augmentation=AugmentationConfig(
                    time_stretch_min=float(params[21]),
                    time_stretch_max=float(params[22]),
                    pitch_shift_min=int(params[23]),
                    pitch_shift_max=int(params[24]),
                    background_noise_prob=float(params[25]),
                    rir_prob=float(params[26]),
                    noise_snr_min=float(params[27]),
                    noise_snr_max=float(params[28]),
                    rir_dry_wet_min=float(params[29]),
                    rir_dry_wet_max=float(params[30]),
                    rir_dry_wet_strategy=str(params[31]),
                    # SpecAugment
                    use_spec_augment=bool(params[32]),
                    freq_mask_param=int(params[33]),
                    time_mask_param=int(params[34]),
                    n_freq_masks=int(params[35]),
                    n_time_masks=int(params[36]),
                    # Time Shift
                    time_shift_prob=float(params[58]),
                    time_shift_min_ms=int(params[59]),
                    time_shift_max_ms=int(params[60]),
                ),
                optimizer=OptimizerConfig(
                    optimizer=params[37],
                    scheduler=params[38],
                    weight_decay=float(params[39]),
                    gradient_clip=float(params[40]),
                    mixed_precision=bool(params[41]),
                    momentum=float(params[42]),
                    warmup_epochs=int(params[43]),
                    min_lr=float(params[44]),
                    step_size=int(params[45]),
                    gamma=float(params[46]),
                    patience=int(params[47]),
                    factor=float(params[48]),
                ),
                loss=LossConfig(
                    loss_function=params[50],
                    label_smoothing=float(params[51]),
                    class_weights=params[52],
                    hard_negative_weight=float(params[53]),
                    focal_alpha=float(params[54]),
                    focal_gamma=float(params[55]),
                    sampler_strategy=params[56],
                    triplet_margin=float(params[61]),
                    class_weight_min=float(params[80]),
                    class_weight_max=float(params[81]),
                ),
                qat=QATConfig(
                    enabled=bool(params[62]), 
                    backend=params[63],
                    start_epoch=int(params[64]),
                ),
                distillation=DistillationConfig(
                    enabled=bool(params[65]), 
                    teacher_architecture=params[66], 
                    temperature=float(params[67]), 
                    alpha=float(params[68])
                )
            )

        def _config_to_params(config: WakewordConfig) -> List:
            """Convert WakewordConfig to UI parameters"""
            return [
                # Data (0-5)
                config.data.sample_rate,
                config.data.audio_duration,
                config.data.n_mfcc,
                config.data.n_fft,
                config.data.hop_length,
                config.data.n_mels,
                # NPY (6-10)
                config.data.use_precomputed_features_for_training,
                config.data.npy_cache_features,
                config.data.fallback_to_audio,
                config.data.npy_feature_dir,
                config.data.npy_feature_type,
                # Training (11-14)
                config.training.batch_size,
                config.training.epochs,
                config.training.learning_rate,
                config.training.early_stopping_patience,
                # Model (15-20)
                config.model.architecture,
                config.model.num_classes,
                config.model.dropout,
                config.model.hidden_size,
                config.model.num_layers,
                config.model.bidirectional,
                # Augmentation (21-31)
                config.augmentation.time_stretch_min,
                config.augmentation.time_stretch_max,
                config.augmentation.pitch_shift_min,
                config.augmentation.pitch_shift_max,
                config.augmentation.background_noise_prob,
                config.augmentation.rir_prob,
                config.augmentation.noise_snr_min,
                config.augmentation.noise_snr_max,
                config.augmentation.rir_dry_wet_min,
                config.augmentation.rir_dry_wet_max,
                config.augmentation.rir_dry_wet_strategy,
                # SpecAugment (32-36)
                config.augmentation.use_spec_augment,
                config.augmentation.freq_mask_param,
                config.augmentation.time_mask_param,
                config.augmentation.n_freq_masks,
                config.augmentation.n_time_masks,
                # Optimizer (37-48)
                config.optimizer.optimizer,
                config.optimizer.scheduler,
                config.optimizer.weight_decay,
                config.optimizer.gradient_clip,
                config.optimizer.mixed_precision,
                config.optimizer.momentum,
                config.optimizer.warmup_epochs,
                config.optimizer.min_lr,
                config.optimizer.step_size,
                config.optimizer.gamma,
                config.optimizer.patience,
                config.optimizer.factor,
                # Workers (49)
                config.training.num_workers,
                # Loss (50-56)
                config.loss.loss_function,
                config.loss.label_smoothing,
                config.loss.class_weights,
                config.loss.hard_negative_weight,
                config.loss.focal_alpha,
                config.loss.focal_gamma,
                config.loss.sampler_strategy,
                # Checkpoint (57)
                config.training.checkpoint_frequency,
                # Time Shift (58-60)
                getattr(config.augmentation, "time_shift_prob", 0.0),
                getattr(config.augmentation, "time_shift_min_ms", -100),
                getattr(config.augmentation, "time_shift_max_ms", 100),
                # Triplet (61)
                getattr(config.loss, "triplet_margin", 1.0),
                # QAT (62-64)
                config.qat.enabled,
                config.qat.backend,
                config.qat.start_epoch,
                # Distillation (65-68)
                config.distillation.enabled,
                config.distillation.teacher_architecture,
                config.distillation.temperature,
                config.distillation.alpha,
                # New Params (69-82)
                getattr(config.training, "use_ema", True),
                getattr(config.training, "ema_decay", 0.999),
                getattr(config.training, "ema_final_decay", 0.9995),
                getattr(config.training, "ema_final_epochs", 10),
                getattr(config.training, "metric_window_size", 100),
                ", ".join(map(str, getattr(config.model, "tcn_num_channels", [64, 128, 256]))),
                getattr(config.model, "tcn_kernel_size", 3),
                getattr(config.model, "tcn_dropout", 0.3),
                ", ".join(map(str, getattr(config.model, "cddnn_hidden_layers", [512, 256, 128]))),
                getattr(config.model, "cddnn_context_frames", 50),
                getattr(config.model, "cddnn_dropout", 0.3),
                getattr(config.loss, "class_weight_min", 0.1),
                getattr(config.loss, "class_weight_max", 100.0),
            ]

        def load_preset_handler(preset_name: str) -> Tuple:
            """Load configuration preset"""
            global _current_config

            try:
                logger.info(f"Loading preset: {preset_name}")

                # Get preset configuration
                config = get_preset(preset_name)
                _current_config = config

                # Update global state if available
                if state is not None:
                    state.value = state.value or {}
                    state.value["config"] = config

                # Convert to UI parameters
                params = _config_to_params(config)

                # Prepare status message
                status = f"✅ Loaded preset: {preset_name}\n{config.description}"

                logger.info(f"Preset loaded successfully: {preset_name}")

                return tuple(params + [status, gr.update(visible=False)])

            except Exception as e:
                error_msg = f"Error loading preset: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return tuple(
                    [None] * len(all_inputs)
                    + [f"❌ {error_msg}", gr.update(visible=False)]
                )

        def save_config_handler(*params) -> str:
            """Save configuration to YAML file"""
            global _current_config

            try:
                # Create config from current parameters
                config = _params_to_config(params)
                _current_config = config

                # Update global state if available
                if state is not None:
                    state.value = state.value or {}
                    state.value["config"] = config

                # Save to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = Path(f"configs/config_{timestamp}.yaml")
                config.save(save_path)

                logger.info(f"Configuration saved to: {save_path}")

                return f"✅ Configuration saved to: {save_path}"

            except Exception as e:
                error_msg = f"Error saving configuration: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return f"❌ {error_msg}"

        def load_config_handler() -> Tuple:
            """Load configuration from YAML file"""
            global _current_config

            try:
                # Find most recent config
                config_dir = Path("configs")
                if not config_dir.exists():
                    return tuple(
                        [None] * len(all_inputs)
                        + ["❌ No saved configurations found", gr.update(visible=False)]
                    )

                config_files = sorted(config_dir.glob("config_*.yaml"), reverse=True)
                if not config_files:
                    return tuple(
                        [None] * len(all_inputs)
                        + ["❌ No saved configurations found", gr.update(visible=False)]
                    )

                # Load most recent
                latest_config = config_files[0]
                logger.info(f"Loading configuration from: {latest_config}")

                config = WakewordConfig.load(latest_config)
                _current_config = config

                # Update global state if available
                if state is not None:
                    state.value = state.value or {}
                    state.value["config"] = config

                # Convert to UI parameters
                params = _config_to_params(config)

                status = f"✅ Loaded configuration from: {latest_config.name}"
                logger.info("Configuration loaded successfully")

                return tuple(params + [status, gr.update(visible=False)])

            except Exception as e:
                error_msg = f"Error loading configuration: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return tuple(
                    [None] * len(all_inputs)
                    + [f"❌ {error_msg}", gr.update(visible=False)]
                )

        def reset_config_handler() -> Tuple:
            """Reset to default configuration"""
            global _current_config

            try:
                logger.info("Resetting to default configuration")

                # Get default config
                config = get_preset("Default")
                _current_config = config

                # Update global state if available
                if state is not None:
                    state.value = state.value or {}
                    state.value["config"] = config

                # Convert to UI parameters
                params = _config_to_params(config)

                status = "✅ Reset to default configuration"
                logger.info("Configuration reset successfully")

                return tuple(params + [status, gr.update(visible=False)])

            except Exception as e:
                error_msg = f"Error resetting configuration: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return tuple(
                    [None] * len(all_inputs)
                    + [f"❌ {error_msg}", gr.update(visible=False)]
                )

        def validate_config_handler(*params) -> Tuple[str, str]:
            """Validate current configuration"""
            try:
                logger.info("Validating configuration")

                # Create config from current parameters
                config = _params_to_config(params)

                # Validate
                validator = ConfigValidator()
                is_valid, issues = validator.validate(config)

                # Generate report
                report = validator.generate_report()

                if is_valid:
                    status = "✅ Configuration is valid and ready for training"
                else:
                    status = f"❌ Configuration has {len([i for i in issues if i.severity == 'error'])} errors"

                logger.info(
                    f"Validation complete: {'valid' if is_valid else 'invalid'}"
                )

                return status, gr.update(value=report, visible=True)

            except WakewordException as e:
                error_msg = f"❌ Configuration Error: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return error_msg, gr.update(
                    value=f"Actionable suggestion: Please check your configuration for the following error: {e}",
                    visible=True,
                )
            except Exception as e:
                error_msg = f"Error validating configuration: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return f"❌ {error_msg}", gr.update(visible=False)

        # Connect event handlers
        load_preset_btn.click(
            fn=load_preset_handler,
            inputs=[preset_dropdown],
            outputs=all_inputs + [config_status, validation_report],
        )

        save_config_btn.click(
            fn=save_config_handler, inputs=all_inputs, outputs=[config_status]
        )

        load_config_btn.click(
            fn=load_config_handler,
            outputs=all_inputs + [config_status, validation_report],
        )

        reset_config_btn.click(
            fn=reset_config_handler,
            outputs=all_inputs + [config_status, validation_report],
        )

        validate_btn.click(
            fn=validate_config_handler,
            inputs=all_inputs,
            outputs=[config_status, validation_report],
        )

    return panel


if __name__ == "__main__":
    # Test the panel
    demo = create_config_panel()
    demo.launch()