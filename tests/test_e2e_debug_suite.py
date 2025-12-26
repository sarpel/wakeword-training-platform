from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.evaluation.types import EvaluationResult
from src.ui.panel_evaluation import (
    EvaluationState,
    collect_false_positives,
    eval_state,
    load_model,
    run_benchmark_test,
    run_threshold_analysis,
)


@pytest.fixture
def mock_eval_state():
    """Reset global eval state before each test"""
    # Save original state
    original_model = eval_state.model
    original_info = eval_state.model_info

    # Reset
    eval_state.model = None
    eval_state.model_info = None
    eval_state.evaluator = None
    eval_state.threshold_analyzer = None
    eval_state.test_results = []

    yield

    # Restore (optional, but good practice)
    eval_state.model = original_model
    eval_state.model_info = original_info


@patch("src.ui.panel_evaluation.load_model_for_evaluation")
def test_full_debug_workflow(mock_load, mock_eval_state):
    """
    Test the complete flow:
    1. Load Model
    2. Run Threshold Analysis (mocked data)
    3. Collect False Positives
    4. Run Benchmark
    """
    # 1. Setup Mock Model
    mock_model = MagicMock()
    mock_config = MagicMock()
    mock_config.data.sample_rate = 16000
    mock_config.data.audio_duration = 1.0
    mock_config.data.n_mels = 64
    mock_config.data.n_mfcc = 13
    mock_config.data.n_fft = 512
    mock_config.data.hop_length = 160
    mock_config.data.feature_type = "mfcc"
    mock_config.model.architecture = "test_arch"

    mock_load.return_value = (mock_model, {"config": mock_config, "epoch": 10, "val_loss": 0.1})

    # Action: Load Model
    status = load_model("dummy/path/model.pt")
    assert "Successfully" in status
    assert eval_state.model is not None

    # 2. Simulate Evaluation Results for Threshold Analysis
    # We need to manually populate eval_state since we aren't running the full dataset eval here
    # (that's covered in unit tests). We just want to check the Integration of the Analysis tool.

    eval_state.threshold_analyzer = MagicMock()
    eval_state.threshold_analyzer.analyze_range.return_value = [
        {"threshold": 0.1, "precision": 0.5, "recall": 1.0},
        {"threshold": 0.9, "precision": 1.0, "recall": 0.5},
    ]

    # Action: Run Analysis
    fig, df = run_threshold_analysis()
    assert fig is not None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2

    # 3. Test False Positive Collection
    # Populate test results manually
    eval_state.test_results = [
        EvaluationResult(
            filename="file1.wav",
            prediction="Positive",
            confidence=0.9,
            label=0,
            raw_audio=np.zeros(16000),
            logits=np.array([0.9]),
            latency_ms=10.0,
        ),  # FP
        EvaluationResult(
            filename="file2.wav",
            prediction="Negative",
            confidence=0.1,
            label=0,
            raw_audio=np.zeros(16000),
            logits=np.array([0.1]),
            latency_ms=10.0,
        ),  # TN
    ]
    eval_state.last_labels = [0, 0]  # Both negative

    # Action: Collect FPs
    html = collect_false_positives()
    assert "file1.wav" in html
    assert "file2.wav" not in html

    # 4. Run Benchmark
    with patch("src.ui.panel_evaluation.BenchmarkRunner") as mock_runner:
        mock_runner.return_value.run_benchmark.return_value = {
            "name": "test_arch",
            "mean_latency_ms": 5.0,
            "process_memory_mb": 10.0,
        }

        results = run_benchmark_test()
        assert results["Model"] == "test_arch"
        assert "5.00 ms" in results["Mean Latency"]
