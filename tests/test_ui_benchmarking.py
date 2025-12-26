from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.ui.panel_evaluation import eval_state, run_benchmark_test


def test_run_benchmark_no_model():
    """Test benchmarking when no model is loaded."""
    eval_state.model = None
    result = run_benchmark_test()
    assert "error" in result
    assert result["error"] == "No model loaded"


@patch("src.ui.panel_evaluation.BenchmarkRunner")
@patch("src.ui.panel_evaluation.SentryInferenceStage")
def test_run_benchmark_success(mock_stage, mock_runner):
    """Test successful benchmark run."""
    # Setup mock model and state
    eval_state.model = MagicMock()
    eval_state.model_info = {
        "config": MagicMock(
            model=MagicMock(architecture="mobilenetv3"), data=MagicMock(sample_rate=16000, audio_duration=1.0)
        )
    }

    # Setup mock runner return value
    mock_runner_instance = mock_runner.return_value
    mock_runner_instance.run_benchmark.return_value = {
        "name": "mobilenetv3",
        "mean_latency_ms": 10.5,
        "min_latency_ms": 9.0,
        "max_latency_ms": 12.0,
        "process_memory_mb": 50.0,
        "gpu_memory_allocated_mb": 100.0,
    }

    result = run_benchmark_test()

    assert result["Model"] == "mobilenetv3"
    assert "10.50 ms" in result["Mean Latency"]
    assert "50.00 MB" in result["RAM Usage"]
