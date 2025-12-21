import pytest
import numpy as np
import time
from src.evaluation.benchmarking import BenchmarkRunner
from src.evaluation.types import StageBase

class MockSlowStage(StageBase):
    def predict(self, audio: np.ndarray) -> dict:
        time.sleep(0.05) # 50ms latency
        return {"confidence": 0.5}
    
    @property
    def name(self) -> str:
        return "slow_stage"

def test_benchmark_runner_latency():
    """Test that BenchmarkRunner accurately measures latency."""
    stage = MockSlowStage()
    runner = BenchmarkRunner(stage)
    
    audio = np.random.randn(16000)
    metrics = runner.run_benchmark(audio, num_iterations=5)
    
    assert metrics["name"] == "slow_stage"
    assert metrics["mean_latency_ms"] >= 50
    assert metrics["iterations"] == 5
    assert "memory_allocated_mb" in metrics

def test_benchmark_runner_invalid_input():
    """Test BenchmarkRunner with invalid input."""
    stage = MockSlowStage()
    runner = BenchmarkRunner(stage)
    
    with pytest.raises(ValueError):
        runner.run_benchmark(None)
