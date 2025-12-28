"""
Environment-aware configuration loader.
Detects OS and hardware to set optimal defaults, allowing overrides via .env.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def load_env_file(dotenv_path: Path) -> Dict[str, str]:
    """Simple manual .env file parser"""
    env_vars: Dict[str, str] = {}
    if not dotenv_path.exists():
        return env_vars

    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip()
    return env_vars


# Load .env into os.environ if not already set
root_env = Path(".env")
if root_env.exists():
    env_data = load_env_file(root_env)
    for k, v in env_data.items():
        if k not in os.environ:
            os.environ[k] = v


class EnvConfig:
    """Helper to resolve environment-specific settings"""

    @property
    def is_windows(self) -> bool:
        return sys.platform == "win32"

    @property
    def is_linux(self) -> bool:
        return sys.platform.startswith("linux")

    @property
    def default_quantization_backend(self) -> str:
        # qnnpack is better for ARM/WSL, fbgemm for native Windows x86
        if self.is_windows:
            return "fbgemm"
        return "qnnpack"

    @property
    def default_mp_start_method(self) -> str:
        # Windows MUST use spawn. Linux can use fork for speed.
        if self.is_windows:
            return "spawn"
        return "fork"

    @property
    def quantization_backend(self) -> str:
        return os.getenv("QUANTIZATION_BACKEND", self.default_quantization_backend)

    @property
    def mp_start_method(self) -> str:
        return os.getenv("MP_START_METHOD", self.default_mp_start_method)

    @property
    def use_triton(self) -> bool:
        # Default False, especially on Windows
        val = os.getenv("USE_TRITON", "false").lower()
        if self.is_windows:
            return False
        return val == "true"

    def get_int(self, key: str, default: int) -> int:
        val = os.getenv(key)
        if val is None:
            return default
        try:
            return int(val)
        except ValueError:
            return default


# Global instance
env_config = EnvConfig()
