"""
Performance benchmarking for inference stages.
"""

import os
import time
from typing import Any, Dict, List

import numpy as np
import psutil
import structlog
import torch

from src.evaluation.types import StageBase

logger = structlog.get_logger(__name__)


class BenchmarkRunner:
    """
    Utility for profiling latency and memory usage of inference stages.
    """

    def __init__(self, stage: StageBase):
        self.stage = stage

    def run_benchmark(self, audio: np.ndarray, num_iterations: int = 10) -> Dict[str, Any]:
        """
        Measure performance metrics for the stage.

        Args:
            audio: Input audio array.
            num_iterations: Number of times to run inference for averaging.

        Returns:
            Dictionary with performance metrics.
        """
        if audio is None or not isinstance(audio, np.ndarray):
            raise ValueError("Input audio must be a numpy array")

        latencies = []

        # Warm-up
        for _ in range(2):
            self.stage.predict(audio)

        # Benchmark loop
        start_mem = self._get_memory_usage()

        for i in range(num_iterations):
            start_time = time.perf_counter()
            self.stage.predict(audio)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)

        end_mem = self._get_memory_usage()

        metrics = {
            "name": self.stage.name,
            "mean_latency_ms": np.mean(latencies),
            "std_latency_ms": np.std(latencies),
            "min_latency_ms": np.min(latencies),
            "max_latency_ms": np.max(latencies),
            "iterations": num_iterations,
            "process_memory_mb": self._get_memory_usage(),
        }

        # Add GPU memory if applicable
        if torch.cuda.is_available():
            metrics["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)

        logger.info(f"Benchmark completed for {self.stage.name}", metrics=metrics)
        return metrics

    def _get_memory_usage(self) -> float:
        """Get current process memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
