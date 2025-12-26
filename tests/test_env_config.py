"""
Test Environment-Aware Configuration
"""

import os
import unittest
from pathlib import Path

from src.config.defaults import QATConfig, TrainingConfig
from src.config.env_config import EnvConfig, env_config


class TestEnvConfig(unittest.TestCase):
    def test_os_detection(self):
        # Basic sanity check
        import sys

        if sys.platform == "win32":
            self.assertTrue(env_config.is_windows)
            self.assertFalse(env_config.is_linux)
        elif sys.platform.startswith("linux"):
            self.assertTrue(env_config.is_linux)
            self.assertFalse(env_config.is_windows)

    def test_defaults_by_os(self):
        # Create a clean instance for testing
        test_env = EnvConfig()

        # We can't easily change sys.platform at runtime for testing,
        # but we can test the logic by mocking the properties if needed.
        # For now, just verify it matches the current OS.
        if test_env.is_windows:
            self.assertEqual(test_env.default_quantization_backend, "fbgemm")
            self.assertEqual(test_env.default_mp_start_method, "spawn")
            self.assertFalse(test_env.use_triton)
        elif test_env.is_linux:
            self.assertEqual(test_env.default_quantization_backend, "qnnpack")
            self.assertEqual(test_env.default_mp_start_method, "fork")

    def test_env_overrides(self):
        # Mock environment variables
        os.environ["QUANTIZATION_BACKEND"] = "test_backend"
        os.environ["MP_START_METHOD"] = "test_method"
        os.environ["USE_TRITON"] = "true"
        os.environ["TRAINING_NUM_WORKERS"] = "99"

        # We must reload the module or use a fresh EnvConfig for the properties
        # to re-evaluate os.getenv if they were cached, but in our implementation
        # they call os.getenv directly in the property.
        test_env = EnvConfig()
        self.assertEqual(test_env.quantization_backend, "test_backend")
        self.assertEqual(test_env.mp_start_method, "test_method")

        # Triton should be False on Windows regardless of env var in our implementation
        # (Safety first)
        import sys

        if sys.platform == "win32":
            self.assertFalse(test_env.use_triton)
        else:
            self.assertTrue(test_env.use_triton)

        self.assertEqual(test_env.get_int("TRAINING_NUM_WORKERS", 8), 99)

    def test_config_integration(self):
        # Since TrainingConfig and QATConfig defaults are evaluated at import time,
        # and they were already imported at the top of this file,
        # we can't easily test the 'dynamic' nature without re-importing.
        # However, for the test to pass in this process, we can check if they
        # match what was in the environment when they WERE imported.

        # For a clean test, we'll use a local import after setting env
        os.environ["TRAINING_NUM_WORKERS"] = "123"
        os.environ["QUANTIZATION_BACKEND"] = "qnnpack_test"

        import importlib

        import src.config.defaults

        importlib.reload(src.config.defaults)
        from src.config.defaults import QATConfig, TrainingConfig

        tc = TrainingConfig()
        self.assertEqual(tc.num_workers, 123)

        qc = QATConfig()
        self.assertEqual(qc.backend, "qnnpack_test")


if __name__ == "__main__":
    unittest.main()
