"""
Tests for StreamingDetector stability features (hysteresis, smoothing).
"""

import pytest

from src.config.defaults import StreamingConfig
from src.evaluation.streaming_detector import StreamingDetector


class TestStreamingStability:
    """Test streaming stability and hysteresis"""

    def test_hysteresis_activation_deactivation(self):
        """Test that hysteresis correctly handles activation and deactivation thresholds"""
        # high=0.7, low=0.3, window=3, threshold=2
        config = StreamingConfig(hysteresis_high=0.7, hysteresis_low=0.3, smoothing_window=3)
        # vote_threshold is calculated as window // 2 + 1 = 2
        detector = StreamingDetector(config=config)

        # 1. Test Activation
        assert detector.step(0.8, 100) is False  # Vote 1
        assert detector.step(0.75, 200) is True  # Vote 2 -> ACTIVE
        assert detector.is_active is True

        # 2. Test stability in "gray zone" (between 0.3 and 0.7)
        # Even if scores drop to 0.4, it should stay active
        assert detector.step(0.4, 3000) is False  # (3000ms is outside lockout of 500ms)
        assert detector.is_active is True

        # 3. Test Deactivation
        # Needs 2 votes below 0.3
        assert detector.step(0.2, 3100) is False
        assert detector.is_active is True  # Still active (only 1 vote below 0.3)
        assert detector.step(0.25, 3200) is False
        assert detector.is_active is False  # DEACTIVATED (2 votes below 0.3)

    def test_smoothing_window_impact(self):
        """Test that larger smoothing windows filter out transient noise"""
        config = StreamingConfig(smoothing_window=10)
        # vote_threshold = 6
        detector = StreamingDetector(config=config)

        # Transient spikes (noisy detections)
        for i in range(5):
            assert detector.step(0.9, i * 100) is False

        assert detector.is_active is False  # Not enough votes (5/10)

        # One more vote
        assert detector.step(0.9, 500) is True  # (6/10) -> ACTIVE
        assert detector.is_active is True

    def test_lockout_period(self):
        """Test that detections are suppressed during lockout"""
        config = StreamingConfig(cooldown_ms=1000, smoothing_window=1)
        detector = StreamingDetector(config=config)

        # First detection
        assert detector.step(0.9, 0) is True

        # Immediate subsequent scores should be ignored
        assert detector.step(0.9, 100) is False
        assert detector.step(0.9, 900) is False

        # After lockout
        # Note: step() returns True on the *moment* of activation.
        # If it was already active and didn't deactivate, it won't return True again.
        # But wait, activation sets locked_until_ms.
        # If it's active, it stays active until votes_off >= threshold.

        # Let's deactivate first
        detector.step(0.1, 1100)
        assert detector.is_active is False

        # Detect again after lockout
        assert detector.step(0.9, 1200) is True
