"""Performance benchmarks for model inference speed."""
import time
from pathlib import Path

import pytest
import torch

from src.models.judge import JudgeModel
from src.models.sentry import SentryModel


@pytest.mark.benchmark
class TestInferenceLatency:
    """Benchmark inference speed against production targets."""

    @pytest.fixture(scope="class")
    def sentry_model(self):
        """Load Sentry model once for all tests."""
        return SentryModel.load_pretrained()

    @pytest.fixture(scope="class")
    def judge_model(self):
        """Load Judge model once for all tests."""
        return JudgeModel.load_pretrained()

    @pytest.fixture
    def test_audio(self):
        """Load test audio sample."""
        audio_path = Path("data/test/wakeword_sample.wav")
        if not audio_path.exists():
            pytest.skip("Test audio not found")
        from src.audio.preprocessing import AudioPreprocessor

        return AudioPreprocessor.load(audio_path)

    def test_sentry_inference_latency(self, sentry_model, test_audio):
        """Sentry inference should be <50ms (edge device target)."""
        # Warmup
        for _ in range(10):
            _ = sentry_model.predict(test_audio)

        # Benchmark
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            _ = sentry_model.predict(test_audio)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        print(f"\nSentry Latency - Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms")

        assert avg_latency < 50, f"Avg latency {avg_latency:.2f}ms exceeds 50ms target"
        assert p95_latency < 100, f"P95 latency {p95_latency:.2f}ms exceeds 100ms"

    def test_judge_inference_latency(self, judge_model, test_audio):
        """Judge inference should be <150ms (local device target)."""
        # Warmup
        for _ in range(10):
            _ = judge_model.predict(test_audio)

        # Benchmark
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            _ = judge_model.predict(test_audio)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        print(f"\nJudge Latency - Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms")

        assert avg_latency < 150, f"Avg latency {avg_latency:.2f}ms exceeds 150ms target"
        assert p95_latency < 200, f"P95 latency {p95_latency:.2f}ms exceeds 200ms"

    def test_cascade_end_to_end_latency(self, sentry_model, judge_model, test_audio):
        """Full cascade should be <200ms total."""
        latencies = []

        for _ in range(100):
            start = time.perf_counter()

            # Stage 1
            sentry_score = sentry_model.predict(test_audio)

            # Stage 2 (conditional)
            if sentry_score > 0.7:
                _ = judge_model.predict(test_audio)

            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        print(f"\nCascade Latency - Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms")

        assert avg_latency < 200, f"Avg latency {avg_latency:.2f}ms exceeds 200ms target"
