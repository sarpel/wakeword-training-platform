"""
Configuration Validator
Validates configuration parameters and checks for compatibility using Pydantic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import structlog

logger = structlog.get_logger(__name__)

# --- Optional deps: WakewordConfig (dataclass-like), Pydantic model, CUDA validator ---

# WakewordConfig: beklenen arayüz sadece to_dict()
try:
    from src.config.schema import WakewordConfig  # gerçek tip burada ise
except Exception:
    from typing import Protocol  # fallback tip

    class WakewordConfigProtocol(Protocol):
        def to_dict(self) -> Dict[str, Any]: ...

    WakewordConfig = WakewordConfigProtocol


# Pydantic model ve sürüm-agnostik doğrulayıcı
WakewordPydanticConfig = None
PydanticValidationError: Any = None
_pydantic_version = 0

try:
    # Önce projedeki gerçek modeli dene
    from src.config.pydantic_validator import WakewordPydanticConfig as _RealModel

    WakewordPydanticConfig = _RealModel
except Exception:
    pass

if WakewordPydanticConfig is None:
    try:
        # Pydantic yüklüyse generic bir doğrulama yolu kur
        pass

        try:
            from pydantic import ValidationError as PydanticValidationError  # v1

            _pydantic_version = 1
        except Exception:
            from pydantic_core import ValidationError as PydanticValidationError  # v2

            _pydantic_version = 2

        # Şema yoksa, Pydantic aşamasını pas geçeceğiz. Ama tipler mevcut.
    except Exception:
        PydanticValidationError = None  # pydantic yok


# CUDA bilgisi
def get_cuda_validator() -> Any:
    """
    Projedeki gerçek get_cuda_validator yoksa basit fallback.
    Beklenen arayüz:
      - .cuda_available: bool
      - .get_memory_info(device_index) -> {'total_gb': float, 'free_gb': float}
    """
    try:
        from src.system.cuda_utils import get_cuda_validator as real_get

        return real_get()
    except Exception:
        # Fallback: torch ile basit bilgi
        try:
            import torch

            class _TorchCudaValidator:
                @property
                def cuda_available(self) -> bool:
                    return torch.cuda.is_available()

                def get_memory_info(self, device_index: int = 0) -> Dict[str, float]:
                    if not torch.cuda.is_available():
                        return {"total_gb": 0.0, "free_gb": 0.0}
                    prop = torch.cuda.get_device_properties(device_index)
                    total_gb = prop.total_memory / (1024**3)
                    # free bilgisi için tahmini bir yaklaşım
                    torch.cuda.synchronize(device_index)
                    # PyTorch doğrudan "free" vermez; kaba bir tahmin:
                    # alloc + reserved üzerinden çıkarım yapmak yerine konservatif davran.
                    return {
                        "total_gb": float(total_gb),
                        "free_gb": max(0.0, float(total_gb) * 0.8),
                    }

            return _TorchCudaValidator()
        except Exception:

            class _NoCuda:
                @property
                def cuda_available(self) -> bool:
                    return False

                def get_memory_info(self, device_index: int = 0) -> Dict[str, float]:
                    return {"total_gb": 0.0, "free_gb": 0.0}

            return _NoCuda()


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
        severity_symbols = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}
        symbol = severity_symbols.get(self.severity, "•")
        return f"{symbol} {self.field}: {self.message}"


class ConfigValidator:
    """Validates wakeword training configuration using Pydantic"""

    def __init__(self) -> None:
        """Initialize validator"""
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
        self.cuda_validator = get_cuda_validator()

    def _run_pydantic_validation(self, config_dict: Dict[str, Any]) -> None:
        """Run Pydantic validation if model or library is available."""
        if WakewordPydanticConfig is not None:
            # Proje-özel Pydantic model varsa, onunla doğrula
            try:
                # v1: parse_obj, v2: model_validate
                if hasattr(WakewordPydanticConfig, "parse_obj"):
                    WakewordPydanticConfig.parse_obj(config_dict)  # type: ignore[attr-defined]
                elif hasattr(WakewordPydanticConfig, "model_validate"):
                    WakewordPydanticConfig.model_validate(config_dict)  # type: ignore[attr-defined]
                else:
                    # Model var ama API belirsiz: attempt construct to catch errors minimally
                    WakewordPydanticConfig(**config_dict)  # type: ignore[call-arg]
            except Exception as e:  # geniş tut, altında hata ayrıştır
                if PydanticValidationError and isinstance(e, PydanticValidationError):
                    for error in e.errors():  # type: ignore[attr-defined]
                        field = ".".join(map(str, error.get("loc", [])))
                        msg = error.get("msg", str(error))
                        self.errors.append(ValidationError(field or "<config>", msg))
                else:
                    self.errors.append(ValidationError("<config>", f"Pydantic validation failed: {e}"))
            return

        # Proje-özel model yoksa ama pydantic yüklüyse yine de atla.
        if PydanticValidationError is not None:
            # Şema olmadığı için doğrulama yapmıyoruz; sadece bilgi
            logger.info(
                "validator.pydantic",
                detail="Pydantic model not found; skipping schema validation.",
            )

    def validate(self, config: WakewordConfig) -> Tuple[bool, List[ValidationError]]:
        """
        Validate complete configuration using Pydantic models.

        Args:
            config: WakewordConfig to validate.

        Returns:
            Tuple of (is_valid, list_of_errors_and_warnings).
        """
        self.errors = []
        self.warnings = []

        # Dataclass -> dict
        try:
            config_dict = config.to_dict()
        except Exception as e:
            self.errors.append(ValidationError("<config>", f"Config to_dict() failed: {e}"))
            return False, self.errors

        # Pydantic validation (opsiyonel)
        self._run_pydantic_validation(config_dict)

        # Extra kurallar
        self._add_custom_warnings(config)
        self._validate_gpu_compatibility(config)

        all_issues = self.errors + self.warnings
        is_valid = len(self.errors) == 0
        return is_valid, all_issues

    def _add_custom_warnings(self, config: WakewordConfig) -> None:
        """Add custom warnings not covered by Pydantic validators"""
        # Sample rate warning
        if 8000 <= config.data.sample_rate < 16000:
            self.warnings.append(
                ValidationError(
                    "data.sample_rate",
                    f"Low sample rate: {config.data.sample_rate}Hz (16000Hz recommended)",
                    "warning",
                )
            )

        # Audio duration warnings
        if 0.5 <= config.data.audio_duration < 1.5:
            self.warnings.append(
                ValidationError(
                    "data.audio_duration",
                    f"Short duration: {config.data.audio_duration}s (1.5-2s recommended)",
                    "warning",
                )
            )
        elif config.data.audio_duration > 5.0:
            self.warnings.append(
                ValidationError(
                    "data.audio_duration",
                    f"Long duration: {config.data.audio_duration}s (may increase memory usage)",
                    "warning",
                )
            )

        # MFCC coefficients warnings
        if hasattr(config.data, "n_mfcc"):
            if 0 < config.data.n_mfcc < 13:
                self.warnings.append(
                    ValidationError(
                        "data.n_mfcc",
                        f"Low MFCC count: {config.data.n_mfcc} (13-40 recommended)",
                        "warning",
                    )
                )
            elif config.data.n_mfcc > 128:
                self.warnings.append(
                    ValidationError(
                        "data.n_mfcc",
                        f"High MFCC count: {config.data.n_mfcc} (may slow training)",
                        "warning",
                    )
                )

        # Batch size warning
        if config.training.batch_size > 256:
            self.warnings.append(
                ValidationError(
                    "training.batch_size",
                    f"Large batch size: {config.training.batch_size} (may cause OOM)",
                    "warning",
                )
            )

    def _validate_gpu_compatibility(self, config: WakewordConfig) -> None:
        """Validate GPU compatibility and estimate memory usage"""
        if not self.cuda_validator.cuda_available:
            # GPU mecburi ise error, değilse info: burada hatayı koruyoruz
            self.errors.append(ValidationError("system.gpu", "GPU not available but required for training"))
            return

        # Tahmini bellek hesabı
        total_samples_per_clip = int(config.data.sample_rate * config.data.audio_duration)
        # 1 kanal, float32 varsayımı
        batch_memory_mb = config.training.batch_size * total_samples_per_clip * 4 / (1024 * 1024)  # float32=4 byte

        mem_info = self.cuda_validator.get_memory_info(0)
        available_memory_mb = float(mem_info.get("free_gb", 0.0)) * 1024.0

        # Kaba tahmin: aktivasyonlar + gradientler + model
        estimated_usage_mb = batch_memory_mb * 3.0 + 200.0  # 200 MB model payı

        if available_memory_mb <= 0:
            self.warnings.append(
                ValidationError(
                    "system.gpu",
                    "Could not read free GPU memory; memory checks are approximate",
                    "warning",
                )
            )
        elif estimated_usage_mb > available_memory_mb:
            self.errors.append(
                ValidationError(
                    "training.batch_size",
                    f"Estimated memory usage ({estimated_usage_mb:.0f}MB) exceeds available GPU memory ({available_memory_mb:.0f}MB)",
                )
            )
        elif estimated_usage_mb > available_memory_mb * 0.8:
            self.warnings.append(
                ValidationError(
                    "training.batch_size",
                    f"High memory usage expected ({estimated_usage_mb:.0f}MB / {available_memory_mb:.0f}MB available)",
                    "warning",
                )
            )

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
    try:
        from src.config.defaults import get_default_config

        print("Configuration Validator Test")
        print("=" * 60)
        config = get_default_config()
        validator = ConfigValidator()
        is_valid, issues = validator.validate(config)
        print(validator.generate_report())
        print("\n✅ Validation test passed" if is_valid else "\n❌ Validation test found errors")
        print("\nValidator test complete")
    except Exception as e:
        print("Self-test skipped:", e)
