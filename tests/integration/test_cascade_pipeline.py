"""Integration tests for distributed cascade architecture.

Tests the complete Sentry -> Judge -> Teacher pipeline with real audio data.
"""
import time
from pathlib import Path

import pytest
import torch

from src.audio.preprocessing import AudioPreprocessor
from src.models.judge import JudgeModel
from src.models.sentry import SentryModel
from src.models.teacher import TeacherModel


class TestCascadeIntegration:
    """Test end-to-end cascade pipeline."""

    @pytest.fixture(scope="class")
    def test_audio_wakeword(self):
        """Load test wakeword audio sample."""
        audio_path = Path("data/test/wakeword_sample.wav")
        if not audio_path.exists():
            pytest.skip(f"Test audio not found: {audio_path}")
        return AudioPreprocessor.load(audio_path)

    @pytest.fixture(scope="class")
    def test_audio_non_wakeword(self):
        """Load test non-wakeword audio sample."""
        audio_path = Path("data/test/non_wakeword_sample.wav")
        if not audio_path.exists():
            pytest.skip(f"Test audio not found: {audio_path}")
        return AudioPreprocessor.load(audio_path)

    def test_sentry_judge_pipeline_positive(self, test_audio_wakeword):
        """Test cascade correctly identifies wakeword."""
        sentry = SentryModel.load_pretrained()
        sentry_score = sentry.predict(test_audio_wakeword)
        assert sentry_score > 0.7, f"Sentry failed: {sentry_score} <= 0.7"

        judge = JudgeModel.load_pretrained()
        judge_score = judge.predict(test_audio_wakeword)
        assert judge_score > 0.9, f"Judge failed: {judge_score} <= 0.9"

    def test_sentry_judge_pipeline_negative(self, test_audio_non_wakeword):
        """Test cascade correctly rejects non-wakeword."""
        sentry = SentryModel.load_pretrained()
        sentry_score = sentry.predict(test_audio_non_wakeword)

        if sentry_score > 0.7:
            judge = JudgeModel.load_pretrained()
            judge_score = judge.predict(test_audio_non_wakeword)
            assert judge_score < 0.5, "Judge should reject non-wakeword"
        else:
            assert sentry_score <= 0.7, "Sentry correctly rejected"

    def test_cascade_latency_target(self, test_audio_wakeword):
        """Ensure cascade meets <200ms latency target."""
        sentry = SentryModel.load_pretrained()
        judge = JudgeModel.load_pretrained()

        start = time.time()
        sentry_score = sentry.predict(test_audio_wakeword)
        if sentry_score > 0.7:
            judge_score = judge.predict(test_audio_wakeword)
        latency_ms = (time.time() - start) * 1000
        assert latency_ms < 200, f"Latency {latency_ms:.1f}ms exceeds 200ms target"

    def test_cascade_power_efficiency(self, test_audio_non_wakeword):
        """Verify Sentry stage filters 90%+ of non-wakewords."""
        sentry = SentryModel.load_pretrained()
        non_wakeword_dir = Path("data/test/non_wakewords")
        if not non_wakeword_dir.exists():
            pytest.skip("Non-wakeword test set not found")

        audio_files = list(non_wakeword_dir.glob("*.wav"))[:100]
        sentry_rejections = 0

        for audio_file in audio_files:
            audio = AudioPreprocessor.load(audio_file)
            score = sentry.predict(audio)
            if score <= 0.7:
                sentry_rejections += 1

        rejection_rate = sentry_rejections / len(audio_files)
        assert rejection_rate >= 0.90, f"Sentry only filtered {rejection_rate:.1%} (target: 90%+)"
