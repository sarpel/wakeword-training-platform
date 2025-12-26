"""
Model Size Calculator for Wakeword Detection
Estimates model size (Flash/RAM) based on configuration.
"""

from typing import Any, Dict, Optional, Tuple

from src.config.defaults import WakewordConfig
from src.config.logger import get_logger
from src.config.platform_constraints import get_platform_constraints
from src.models.architectures import create_model

logger = get_logger(__name__)


class SizeCalculator:
    """Calculates and validates model size against platform constraints"""

    def __init__(self, config: WakewordConfig):
        """
        Initialize calculator with configuration

        Args:
            config: WakewordConfig instance
        """
        self.config = config

    def estimate_parameters(self) -> int:
        """
        Estimate number of parameters in the model

        Returns:
            Total number of parameters
        """
        try:
            # Create a dummy model to count parameters
            # We use a try-except because some models might require specific kwargs
            model_params = self.config.model.to_dict()
            arch = model_params.pop("architecture")
            num_classes = model_params.pop("num_classes", 2)
            pretrained = model_params.pop("pretrained", False)

            model = create_model(
                architecture=arch,
                num_classes=num_classes,
                pretrained=pretrained,
                input_channels=1,  # Default for spectrograms
                input_size=self.config.data.n_mels or self.config.data.n_mfcc or 64,
                **model_params,
            )

            total_params = sum(p.numel() for p in model.parameters())
            return total_params
        except Exception as e:
            logger.exception(f"Error creating model for parameter estimation: {e}")
            return 0

    def calculate_estimated_size_kb(self) -> Tuple[float, float]:
        """
        Calculate estimated Flash and RAM usage in KB.

        Returns:
            Tuple of (flash_kb, ram_kb)
        """
        params = self.estimate_parameters()
        if params == 0:
            return 0.0, 0.0

        # Flash estimation
        # FP32 = 4 bytes, INT8 = 1 byte
        bytes_per_param = 1 if self.config.qat.enabled else 4

        # Add ~10% overhead for model metadata/structure (TFLite/ONNX)
        overhead_factor = 1.1
        flash_kb = (params * bytes_per_param * overhead_factor) / 1024

        # RAM estimation (Rule of thumb)
        # RAM usage = Weights (if copied to RAM) + Activations + Audio Buffers
        # For microcontrollers, weights are often in Flash (XIP),
        # but let's assume worst case where some weights or buffers are in RAM.
        # Activation size depends on architecture and input size.

        # Simple heuristic for activations: ~2x the largest layer or ~20% of model size
        activations_kb = (params * 0.2 * 4) / 1024  # Assuming FP32 activations

        # Audio buffer (e.g., 1.5s @ 16kHz)
        audio_buffer_kb = (self.config.data.audio_duration * self.config.data.sample_rate * 4) / 1024

        ram_kb = activations_kb + audio_buffer_kb

        # If QAT is NOT enabled, we might need more RAM for FP32 weights if they don't fit in Flash
        if not self.config.qat.enabled:
            ram_kb += flash_kb * 0.5  # Assume some weights need to be in RAM

        return flash_kb, ram_kb

    def compare_with_platform(self, platform_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare estimated size with platform constraints.

        Args:
            platform_name: Optional name of the platform to compare against.
                          If None, uses size_targets from config.

        Returns:
            Dictionary with comparison results
        """
        flash_est, ram_est = self.calculate_estimated_size_kb()

        max_flash = self.config.size_targets.max_flash_kb
        max_ram = self.config.size_targets.max_ram_kb
        p_name = "Custom Targets"

        if platform_name:
            try:
                constraints = get_platform_constraints(platform_name)
                max_flash = constraints.max_flash_kb
                max_ram = constraints.max_ram_kb
                p_name = constraints.name
            except ValueError:
                logger.warning(f"Platform {platform_name} not found, using config targets.")

        results = {
            "platform_name": p_name,
            "estimated_flash_kb": round(flash_est, 2),
            "estimated_ram_kb": round(ram_est, 2),
            "max_flash_kb": max_flash,
            "max_ram_kb": max_ram,
            "flash_ok": flash_est <= max_flash if max_flash > 0 else True,
            "ram_ok": ram_est <= max_ram if max_ram > 0 else True,
            "params_count": self.estimate_parameters(),
        }

        return results

    def get_summary_report(self, platform_name: Optional[str] = None) -> str:
        """Generate a human-readable summary report"""
        res = self.compare_with_platform(platform_name)

        report = [
            f"--- Model Size Insight: {res['platform_name']} ---",
            f"Architecture: {self.config.model.architecture}",
            f"Parameters: {res['params_count']:,}",
            f"Quantization: {'Enabled (INT8)' if self.config.qat.enabled else 'Disabled (FP32)'}",
            f"Estimated Flash: {res['estimated_flash_kb']} KB / {res['max_flash_kb'] or '∞'} KB "
            + f"({'✅' if res['flash_ok'] else '❌'})",
            f"Estimated RAM:   {res['estimated_ram_kb']} KB / {res['max_ram_kb'] or '∞'} KB "
            + f"({'✅' if res['ram_ok'] else '❌'})",
        ]

        if not res["flash_ok"]:
            report.append("⚠️ WARNING: Model might be too large for Flash!")
        if not res["ram_ok"]:
            report.append("⚠️ WARNING: Model might exceed available RAM!")

        return "\n".join(report)


def validate_config_size(config: WakewordConfig, platform_name: Optional[str] = None) -> bool:
    """
    Validate if the current config fits the platform.

    Args:
        config: WakewordConfig
        platform_name: Optional platform name

    Returns:
        True if it fits, False otherwise
    """
    calc = SizeCalculator(config)
    results = calc.compare_with_platform(platform_name)
    return results["flash_ok"] and results["ram_ok"]
