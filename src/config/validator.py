"""
Configuration Validator
Validates configuration parameters and checks for compatibility
"""
from typing import List, Tuple, Dict, Any
import logging
from src.config.defaults import WakewordConfig
from src.config.cuda_utils import get_cuda_validator

logger = logging.getLogger(__name__)


class ValidationError:
    """Validation error with severity"""

    def __init__(self, field: str, message: str, severity: str = "error"):
        """
        Initialize validation error

        Args:
            field: Configuration field name
            message: Error message
            severity: error, warning, or info
        """
        self.field = field
        self.message = message
        self.severity = severity

    def __str__(self) -> str:
        severity_symbols = {
            "error": "❌",
            "warning": "⚠️",
            "info": "ℹ️"
        }
        symbol = severity_symbols.get(self.severity, "•")
        return f"{symbol} {self.field}: {self.message}"


class ConfigValidator:
    """Validates wakeword training configuration"""

    def __init__(self):
        """Initialize validator"""
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
        self.cuda_validator = get_cuda_validator()

    def validate(self, config: WakewordConfig) -> Tuple[bool, List[ValidationError]]:
        """
        Validate complete configuration

        Args:
            config: WakewordConfig to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        self.errors = []
        self.warnings = []

        # Validate each section
        self._validate_data_config(config.data)
        self._validate_training_config(config.training)
        self._validate_model_config(config.model)
        self._validate_augmentation_config(config.augmentation)
        self._validate_optimizer_config(config.optimizer)
        self._validate_loss_config(config.loss)

        # Cross-validation
        self._validate_cross_dependencies(config)

        # GPU-specific validation
        self._validate_gpu_compatibility(config)

        all_issues = self.errors + self.warnings
        is_valid = len(self.errors) == 0

        return is_valid, all_issues

    def _validate_data_config(self, data_config):
        """Validate data configuration"""
        # Sample rate
        if data_config.sample_rate < 8000:
            self.errors.append(ValidationError(
                "data.sample_rate",
                f"Sample rate too low: {data_config.sample_rate}Hz (minimum: 8000Hz)"
            ))
        elif data_config.sample_rate < 16000:
            self.warnings.append(ValidationError(
                "data.sample_rate",
                f"Low sample rate: {data_config.sample_rate}Hz (16000Hz recommended)",
                "warning"
            ))

        # Audio duration
        if data_config.audio_duration <= 0:
            self.errors.append(ValidationError(
                "data.audio_duration",
                "Audio duration must be positive"
            ))
        elif data_config.audio_duration < 0.5:
            self.warnings.append(ValidationError(
                "data.audio_duration",
                f"Very short duration: {data_config.audio_duration}s (1.5-2s recommended)",
                "warning"
            ))
        elif data_config.audio_duration > 5.0:
            self.warnings.append(ValidationError(
                "data.audio_duration",
                f"Long duration: {data_config.audio_duration}s (may increase memory usage)",
                "warning"
            ))

        # MFCC coefficients
        if data_config.n_mfcc < 13:
            self.warnings.append(ValidationError(
                "data.n_mfcc",
                f"Low MFCC count: {data_config.n_mfcc} (13-40 recommended)",
                "warning"
            ))
        elif data_config.n_mfcc > 128:
            self.warnings.append(ValidationError(
                "data.n_mfcc",
                f"High MFCC count: {data_config.n_mfcc} (may slow training)",
                "warning"
            ))

        # FFT size
        valid_fft_sizes = [256, 512, 1024, 2048, 4096]
        if data_config.n_fft not in valid_fft_sizes:
            self.warnings.append(ValidationError(
                "data.n_fft",
                f"Unusual FFT size: {data_config.n_fft} (typical: {valid_fft_sizes})",
                "warning"
            ))

        # Hop length should be less than n_fft
        if data_config.hop_length >= data_config.n_fft:
            self.errors.append(ValidationError(
                "data.hop_length",
                f"Hop length ({data_config.hop_length}) must be less than n_fft ({data_config.n_fft})"
            ))

        # Feature type
        valid_features = ["mel", "mel_spectrogram", "mfcc"]  # mel_spectrogram is legacy, normalized to mel
        if data_config.feature_type not in valid_features:
            self.errors.append(ValidationError(
                "data.feature_type",
                f"Invalid feature type: {data_config.feature_type} (valid: {valid_features})"
            ))

    def _validate_training_config(self, training_config):
        """Validate training configuration"""
        # Batch size
        if training_config.batch_size < 1:
            self.errors.append(ValidationError(
                "training.batch_size",
                "Batch size must be at least 1"
            ))
        elif training_config.batch_size > 256:
            self.warnings.append(ValidationError(
                "training.batch_size",
                f"Large batch size: {training_config.batch_size} (may cause OOM)",
                "warning"
            ))

        # Epochs
        if training_config.epochs < 1:
            self.errors.append(ValidationError(
                "training.epochs",
                "Epochs must be at least 1"
            ))
        elif training_config.epochs < 10:
            self.warnings.append(ValidationError(
                "training.epochs",
                f"Few epochs: {training_config.epochs} (30+ recommended)",
                "warning"
            ))

        # Learning rate
        if training_config.learning_rate <= 0:
            self.errors.append(ValidationError(
                "training.learning_rate",
                "Learning rate must be positive"
            ))
        elif training_config.learning_rate > 0.1:
            self.warnings.append(ValidationError(
                "training.learning_rate",
                f"Very high learning rate: {training_config.learning_rate} (may diverge)",
                "warning"
            ))
        elif training_config.learning_rate < 1e-6:
            self.warnings.append(ValidationError(
                "training.learning_rate",
                f"Very low learning rate: {training_config.learning_rate} (training may be slow)",
                "warning"
            ))

        # Early stopping patience
        if training_config.early_stopping_patience < 1:
            self.errors.append(ValidationError(
                "training.early_stopping_patience",
                "Early stopping patience must be at least 1"
            ))

        # Num workers
        if training_config.num_workers < 0:
            self.errors.append(ValidationError(
                "training.num_workers",
                "Number of workers cannot be negative"
            ))
        elif training_config.num_workers > 16:
            self.warnings.append(ValidationError(
                "training.num_workers",
                f"Many workers: {training_config.num_workers} (may not improve speed)",
                "warning"
            ))

        # Checkpoint frequency
        valid_frequencies = ["best_only", "every_epoch", "every_5_epochs", "every_10_epochs"]
        if training_config.checkpoint_frequency not in valid_frequencies:
            self.errors.append(ValidationError(
                "training.checkpoint_frequency",
                f"Invalid checkpoint frequency: {training_config.checkpoint_frequency} (valid: {valid_frequencies})"
            ))

    def _validate_model_config(self, model_config):
        """Validate model configuration"""
        # Architecture
        valid_architectures = ["resnet18", "mobilenetv3", "lstm", "gru", "tcn"]
        if model_config.architecture not in valid_architectures:
            self.errors.append(ValidationError(
                "model.architecture",
                f"Invalid architecture: {model_config.architecture} (valid: {valid_architectures})"
            ))

        # Number of classes
        if model_config.num_classes < 2:
            self.errors.append(ValidationError(
                "model.num_classes",
                "Number of classes must be at least 2"
            ))

        # Dropout
        if model_config.dropout < 0 or model_config.dropout >= 1:
            self.errors.append(ValidationError(
                "model.dropout",
                f"Dropout must be in [0, 1): {model_config.dropout}"
            ))

        # LSTM/GRU specific
        if model_config.architecture in ["lstm", "gru"]:
            if model_config.hidden_size < 16:
                self.warnings.append(ValidationError(
                    "model.hidden_size",
                    f"Small hidden size: {model_config.hidden_size} (64-256 typical)",
                    "warning"
                ))

            if model_config.num_layers < 1:
                self.errors.append(ValidationError(
                    "model.num_layers",
                    "Number of layers must be at least 1"
                ))

    def _validate_augmentation_config(self, aug_config):
        """Validate augmentation configuration"""
        # Time stretch
        if aug_config.time_stretch_min >= aug_config.time_stretch_max:
            self.errors.append(ValidationError(
                "augmentation.time_stretch",
                f"min ({aug_config.time_stretch_min}) must be less than max ({aug_config.time_stretch_max})"
            ))

        if aug_config.time_stretch_min < 0.5 or aug_config.time_stretch_max > 2.0:
            self.warnings.append(ValidationError(
                "augmentation.time_stretch",
                "Extreme time stretch range (0.5-2.0 recommended)",
                "warning"
            ))

        # Pitch shift
        if aug_config.pitch_shift_min >= aug_config.pitch_shift_max:
            self.errors.append(ValidationError(
                "augmentation.pitch_shift",
                f"min ({aug_config.pitch_shift_min}) must be less than max ({aug_config.pitch_shift_max})"
            ))

        # Pitch shift must be integers (for random.randint)
        if not isinstance(aug_config.pitch_shift_min, int) or not isinstance(aug_config.pitch_shift_max, int):
            self.errors.append(ValidationError(
                "augmentation.pitch_shift",
                f"pitch_shift values must be integers (got min={type(aug_config.pitch_shift_min).__name__}, max={type(aug_config.pitch_shift_max).__name__})"
            ))

        # Probabilities
        for prob_field, prob_value in [
            ("background_noise_prob", aug_config.background_noise_prob),
            ("rir_prob", aug_config.rir_prob),
        ]:
            if not (0 <= prob_value <= 1):
                self.errors.append(ValidationError(
                    f"augmentation.{prob_field}",
                    f"Probability must be in [0, 1]: {prob_value}"
                ))

        # SNR range
        if aug_config.noise_snr_min >= aug_config.noise_snr_max:
            self.errors.append(ValidationError(
                "augmentation.noise_snr",
                f"min ({aug_config.noise_snr_min}) must be less than max ({aug_config.noise_snr_max})"
            ))

        # SpecAugment
        if aug_config.use_spec_augment:
            if not isinstance(aug_config.freq_mask_param, int) or aug_config.freq_mask_param <= 0:
                self.errors.append(ValidationError(
                    "augmentation.freq_mask_param",
                    f"freq_mask_param must be a positive integer, but got {aug_config.freq_mask_param}"
                ))
            if not isinstance(aug_config.time_mask_param, int) or aug_config.time_mask_param <= 0:
                self.errors.append(ValidationError(
                    "augmentation.time_mask_param",
                    f"time_mask_param must be a positive integer, but got {aug_config.time_mask_param}"
                ))
            if not isinstance(aug_config.n_freq_masks, int) or aug_config.n_freq_masks < 0:
                self.errors.append(ValidationError(
                    "augmentation.n_freq_masks",
                    f"n_freq_masks must be a non-negative integer, but got {aug_config.n_freq_masks}"
                ))
            if not isinstance(aug_config.n_time_masks, int) or aug_config.n_time_masks < 0:
                self.errors.append(ValidationError(
                    "augmentation.n_time_masks",
                    f"n_time_masks must be a non-negative integer, but got {aug_config.n_time_masks}"
                ))

    def _validate_optimizer_config(self, opt_config):
        """Validate optimizer configuration"""
        # Optimizer type
        valid_optimizers = ["adam", "sgd", "adamw"]
        if opt_config.optimizer not in valid_optimizers:
            self.errors.append(ValidationError(
                "optimizer.optimizer",
                f"Invalid optimizer: {opt_config.optimizer} (valid: {valid_optimizers})"
            ))

        # Weight decay
        if opt_config.weight_decay < 0:
            self.errors.append(ValidationError(
                "optimizer.weight_decay",
                "Weight decay cannot be negative"
            ))

        # Momentum
        if not (0 <= opt_config.momentum < 1):
            self.errors.append(ValidationError(
                "optimizer.momentum",
                f"Momentum must be in [0, 1): {opt_config.momentum}"
            ))

        # Scheduler
        valid_schedulers = ["cosine", "step", "plateau", "none"]
        if opt_config.scheduler not in valid_schedulers:
            self.errors.append(ValidationError(
                "optimizer.scheduler",
                f"Invalid scheduler: {opt_config.scheduler} (valid: {valid_schedulers})"
            ))

        # Gradient clip
        if opt_config.gradient_clip <= 0:
            self.errors.append(ValidationError(
                "optimizer.gradient_clip",
                "Gradient clip must be positive"
            ))

    def _validate_loss_config(self, loss_config):
        """Validate loss configuration"""
        # Loss function
        valid_losses = ["cross_entropy", "focal_loss"]
        if loss_config.loss_function not in valid_losses:
            self.errors.append(ValidationError(
                "loss.loss_function",
                f"Invalid loss function: {loss_config.loss_function} (valid: {valid_losses})"
            ))

        # Label smoothing
        if not (0 <= loss_config.label_smoothing < 1):
            self.errors.append(ValidationError(
                "loss.label_smoothing",
                f"Label smoothing must be in [0, 1): {loss_config.label_smoothing}"
            ))

        # Focal loss parameters
        if loss_config.loss_function == "focal_loss":
            if not (0 <= loss_config.focal_alpha <= 1):
                self.errors.append(ValidationError(
                    "loss.focal_alpha",
                    f"Focal alpha must be in [0, 1]: {loss_config.focal_alpha}"
                ))

            if loss_config.focal_gamma < 0:
                self.errors.append(ValidationError(
                    "loss.focal_gamma",
                    "Focal gamma must be non-negative"
                ))

        # Class weights
        valid_class_weights = ["balanced", "none", "custom"]
        if loss_config.class_weights not in valid_class_weights:
            self.errors.append(ValidationError(
                "loss.class_weights",
                f"Invalid class weights: {loss_config.class_weights} (valid: {valid_class_weights})"
            ))

        # Hard negative weight
        if loss_config.hard_negative_weight < 1.0:
            self.warnings.append(ValidationError(
                "loss.hard_negative_weight",
                f"Hard negative weight < 1.0: {loss_config.hard_negative_weight} (reduces importance)",
                "warning"
            ))

        # Sampler strategy
        valid_samplers = ["weighted", "balanced", "none"]
        if loss_config.sampler_strategy not in valid_samplers:
            self.errors.append(ValidationError(
                "loss.sampler_strategy",
                f"Invalid sampler: {loss_config.sampler_strategy} (valid: {valid_samplers})"
            ))

    def _validate_cross_dependencies(self, config: WakewordConfig):
        """Validate cross-parameter dependencies"""
        # Check if early stopping patience is reasonable compared to epochs
        if config.training.early_stopping_patience > config.training.epochs / 2:
            self.warnings.append(ValidationError(
                "training.early_stopping_patience",
                f"Patience ({config.training.early_stopping_patience}) is > half of epochs ({config.training.epochs})",
                "warning"
            ))

        # Check warmup epochs vs total epochs
        if config.optimizer.warmup_epochs > config.training.epochs:
            self.errors.append(ValidationError(
                "optimizer.warmup_epochs",
                f"Warmup epochs ({config.optimizer.warmup_epochs}) exceeds total epochs ({config.training.epochs})"
            ))

    def _validate_gpu_compatibility(self, config: WakewordConfig):
        """Validate GPU compatibility and estimate memory usage"""
        if not self.cuda_validator.cuda_available:
            self.errors.append(ValidationError(
                "system.gpu",
                "GPU not available but required for training"
            ))
            return

        # Estimate memory usage
        samples_per_second = config.data.sample_rate * config.data.audio_duration
        batch_memory_mb = (
            config.training.batch_size *
            samples_per_second *
            4 /  # 4 bytes per float32
            (1024 * 1024)
        )

        # Get GPU memory
        mem_info = self.cuda_validator.get_memory_info(0)
        available_memory_mb = mem_info['free_gb'] * 1024

        # Rough estimate: batch + model + gradients (3x batch size)
        estimated_usage_mb = batch_memory_mb * 3 + 200  # 200MB for model

        if estimated_usage_mb > available_memory_mb:
            self.errors.append(ValidationError(
                "training.batch_size",
                f"Estimated memory usage ({estimated_usage_mb:.0f}MB) exceeds available GPU memory ({available_memory_mb:.0f}MB)"
            ))
        elif estimated_usage_mb > available_memory_mb * 0.8:
            self.warnings.append(ValidationError(
                "training.batch_size",
                f"High memory usage expected ({estimated_usage_mb:.0f}MB / {available_memory_mb:.0f}MB available)",
                "warning"
            ))

    def generate_report(self) -> str:
        """
        Generate validation report

        Returns:
            Formatted validation report
        """
        report = []
        report.append("=" * 60)
        report.append("CONFIGURATION VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")

        if not self.errors and not self.warnings:
            report.append("✅ All validation checks passed")
            report.append("Configuration is ready for training")
        else:
            if self.errors:
                report.append(f"❌ Errors: {len(self.errors)}")
                report.append("-" * 60)
                for error in self.errors:
                    report.append(f"  {error}")
                report.append("")

            if self.warnings:
                report.append(f"⚠️  Warnings: {len(self.warnings)}")
                report.append("-" * 60)
                for warning in self.warnings:
                    report.append(f"  {warning}")
                report.append("")

            if self.errors:
                report.append("❌ Configuration has errors and cannot be used")
                report.append("Please fix errors before proceeding")
            else:
                report.append("⚠️  Configuration has warnings but is usable")
                report.append("Consider addressing warnings for optimal results")

        report.append("=" * 60)
        return "\n".join(report)


if __name__ == "__main__":
    # Test validator
    from src.config.defaults import get_default_config

    print("Configuration Validator Test")
    print("=" * 60)

    # Test with default config
    config = get_default_config()
    validator = ConfigValidator()
    is_valid, issues = validator.validate(config)

    print(validator.generate_report())

    if is_valid:
        print("\n✅ Validation test passed")
    else:
        print("\n❌ Validation test found errors")

    print("\nValidator test complete")