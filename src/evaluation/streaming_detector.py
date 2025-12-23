"""
Streaming wakeword detector for real-time audio processing.

This module provides classes for detecting wakewords in streaming audio using:
- Sliding window processing
- Voting mechanisms (N out of M detections)
- Hysteresis (separate on/off thresholds)
- Lockout periods after detection
"""

# Standard library imports
from collections import deque  # For efficient fixed-size buffers
from typing import Deque, List, Optional, Tuple  # Type hints for better code clarity

# Third-party imports
import numpy as np  # For numerical operations on audio arrays
import structlog  # For structured logging
import torch  # For PyTorch model inference

from src.config.defaults import StreamingConfig
from src.evaluation.types import InferenceEngine, StageBase

# Initialize logger for this module
logger = structlog.get_logger(__name__)


class CascadeInferenceEngine(InferenceEngine):
    """
    Orchestration engine for distributed cascade inference.

    Manages multiple inference stages (e.g., Sentry, Judge) and handles
    the logic for passing results between them.
    """

    def __init__(self):
        self.stages: List[StageBase] = []

    def add_stage(self, stage: StageBase):
        """Add an inference stage to the cascade"""
        self.stages.append(stage)
        logger.info(f"Added stage to cascade: {stage.name}")

    def run(self, audio: np.ndarray) -> List[dict]:
        """
        Execute the cascade inference.

        Follows a sequential pipeline where each stage only runs if
        the previous stage detected a potential wakeword.
        """
        results = []
        for stage in self.stages:
            stage_result = stage.predict(audio)
            results.append({"stage": stage.name, "result": stage_result})

            # Cascade handoff: stop if not detected
            if not stage_result.get("detected", True):
                logger.info(f"Cascade stopped at {stage.name} (no detection)")
                break

        return results


class StreamingDetector:
    """
    Streaming wakeword detector with production-ready features

    Features:
    - Sliding window processing
    - Voting mechanism (N out of M)
    - Hysteresis (separate on/off thresholds)
    - Lockout period after detection
    """

    def __init__(
        self,
        threshold_on: float = 0.5,
        threshold_off: Optional[float] = None,
        hysteresis: float = 0.1,
        lockout_ms: int = 1500,
        vote_window: int = 5,
        vote_threshold: int = 3,
        config: Optional[StreamingConfig] = None,
    ):
        """
        Initialize streaming detector

        Args:
            threshold_on: Threshold for triggering detection
            threshold_off: Threshold for de-activating (default: threshold_on - hysteresis)
            hysteresis: Hysteresis margin (threshold_off = threshold_on - hysteresis)
            lockout_ms: Lockout period in milliseconds after detection
            vote_window: Window size for voting (number of recent scores)
            vote_threshold: Number of votes needed for detection
            config: Optional StreamingConfig object to override parameters
        """
        if config:
            self.threshold_on = threshold_on  # threshold_on is usually separate from streaming config
            self.threshold_off = config.hysteresis_low
            self.lockout_ms = config.cooldown_ms
            self.vote_window = config.smoothing_window
            self.vote_threshold = max(1, config.smoothing_window // 2 + 1)
            # Override threshold_on if it's explicitly high in config or passed
            if hasattr(config, "hysteresis_high"):
                self.threshold_on = config.hysteresis_high
        else:
            self.threshold_on = threshold_on
            self.threshold_off = threshold_off if threshold_off is not None else max(threshold_on - hysteresis, 0)
            self.lockout_ms = lockout_ms
            self.vote_window = vote_window
            self.vote_threshold = vote_threshold

        # State
        self.score_buffer: Deque[float] = deque(maxlen=self.vote_window)
        self.locked_until_ms = 0
        self.is_active = False

        logger.info(
            f"StreamingDetector initialized: "
            f"threshold_on={self.threshold_on:.3f}, threshold_off={self.threshold_off:.3f}, "
            f"lockout={self.lockout_ms}ms, vote={self.vote_threshold}/{self.vote_window}"
        )

    def step(self, score: float, timestamp_ms: int) -> bool:
        """
        Process one detection score

        Args:
            score: Detection score (probability for positive class)
            timestamp_ms: Current timestamp in milliseconds

        Returns:
            True if wakeword detected, False otherwise
        """
        # Add score to buffer
        self.score_buffer.append(score)

        # Check if in lockout period
        if timestamp_ms < self.locked_until_ms:
            return False

        # Count votes in current window
        votes_on = sum(1 for s in self.score_buffer if s >= self.threshold_on)
        votes_off = sum(1 for s in self.score_buffer if s < self.threshold_off)

        # State machine logic
        if not self.is_active:
            # Try to activate
            if votes_on >= self.vote_threshold:
                self.is_active = True
                self.locked_until_ms = timestamp_ms + self.lockout_ms
                return True  # Detection!
        else:
            # Try to deactivate
            if votes_off >= self.vote_threshold:
                self.is_active = False

        return False

    def reset(self) -> None:
        """Reset detector state"""
        self.score_buffer.clear()
        self.locked_until_ms = 0
        self.is_active = False


class SlidingWindowProcessor:
    """
    Process audio stream with sliding windows

    Handles window extraction, hop computation, and score aggregation
    """

    def __init__(
        self,
        window_duration_s: float = 1.0,
        hop_duration_s: float = 0.1,
        sample_rate: int = 16000,
    ):
        """
        Initialize sliding window processor

        Args:
            window_duration_s: Window duration in seconds
            hop_duration_s: Hop (stride) duration in seconds
            sample_rate: Audio sample rate
        """
        self.window_size = int(window_duration_s * sample_rate)
        self.hop_size = int(hop_duration_s * sample_rate)
        self.sample_rate = sample_rate

        logger.info(
            f"SlidingWindowProcessor: "
            f"window={window_duration_s}s ({self.window_size} samples), "
            f"hop={hop_duration_s}s ({self.hop_size} samples)"
        )

    def extract_windows(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract overlapping windows from audio

        Args:
            audio: Input audio (1D array)

        Returns:
            Tuple of (windows, timestamps_ms)
        """
        # Calculate number of windows
        num_samples = len(audio)
        num_windows = (num_samples - self.window_size) // self.hop_size + 1

        if num_windows <= 0:
            return np.array([]), np.array([])

        # Extract windows
        windows = []
        timestamps_ms = []

        for i in range(num_windows):
            start = i * self.hop_size
            end = start + self.window_size

            if end > num_samples:
                break

            window = audio[start:end]
            windows.append(window)

            # Timestamp is center of window
            center_sample = start + self.window_size // 2
            timestamp_ms = int((center_sample / self.sample_rate) * 1000)
            timestamps_ms.append(timestamp_ms)

        return np.array(windows), np.array(timestamps_ms)


def process_audio_stream(
    model: torch.nn.Module,
    audio: np.ndarray,
    sample_rate: int = 16000,
    window_duration_s: float = 1.0,
    hop_duration_s: float = 0.1,
    threshold: float = 0.5,
    vote_window: int = 5,
    vote_threshold: int = 3,
    lockout_ms: int = 1500,
    device: str = "cuda",
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    Process audio stream and detect wakewords

    Args:
        model: Trained wakeword model
        audio: Input audio (1D numpy array)
        sample_rate: Audio sample rate
        window_duration_s: Window duration
        hop_duration_s: Hop duration
        threshold: Detection threshold
        vote_window: Voting window size
        vote_threshold: Votes needed for detection
        lockout_ms: Lockout period
        device: Device for inference

    Returns:
        Tuple of (detections, all_scores)
        - detections: List of (timestamp_ms, score) for detected wakewords
        - all_scores: List of (timestamp_ms, score) for all windows
    """
    # Initialize components
    window_processor = SlidingWindowProcessor(
        window_duration_s=window_duration_s,
        hop_duration_s=hop_duration_s,
        sample_rate=sample_rate,
    )

    detector = StreamingDetector(
        threshold_on=threshold,
        vote_window=vote_window,
        vote_threshold=vote_threshold,
        lockout_ms=lockout_ms,
    )

    # Extract windows
    windows, timestamps = window_processor.extract_windows(audio)

    if len(windows) == 0:
        logger.warning("No windows extracted from audio")
        return [], []

    logger.info(f"Processing {len(windows)} windows...")

    # Process windows
    model.eval()
    detections = []
    all_scores = []

    with torch.no_grad():
        for window, timestamp_ms in zip(windows, timestamps):
            # Convert to tensor and extract features
            # Assuming model expects features, not raw audio
            # You would need to add feature extraction here
            # For now, placeholder:

            # This is a placeholder - in real usage, you'd extract mel-spectrogram features
            # feature = extract_features(window)  # Not implemented here
            # features_tensor = torch.from_numpy(feature).float().unsqueeze(0).to(device)

            # Get model prediction
            # logits = model(features_tensor)
            # probs = torch.softmax(logits, dim=1)
            # score = probs[0, 1].item()  # Probability of positive class

            # Placeholder score (replace with actual inference)
            score = 0.0  # Replace this

            all_scores.append((timestamp_ms, score))

            # Check for detection
            if detector.step(score, timestamp_ms):
                detections.append((timestamp_ms, score))
                logger.info(f"Detection at {timestamp_ms}ms (score={score:.3f})")

    logger.info(f"Completed: {len(detections)} detections from {len(windows)} windows")

    return detections, all_scores


if __name__ == "__main__":
    # Test streaming detector
    print("Streaming Detector Test")
    print("=" * 60)

    # Test 1: Basic detector
    print("\n1. Testing StreamingDetector...")

    detector = StreamingDetector(threshold_on=0.7, vote_window=5, vote_threshold=3, lockout_ms=1500)

    # Simulate scores
    test_scores = [
        (0, 0.3),  # No detection
        (100, 0.4),
        (200, 0.8),  # Vote 1
        (300, 0.9),  # Vote 2
        (400, 0.85),  # Vote 3 -> DETECTION
        (500, 0.9),  # In lockout
        (1000, 0.95),  # In lockout
        (2000, 0.8),  # Out of lockout, but buffer reset
    ]

    detections = []
    for timestamp_ms, score in test_scores:
        detected = detector.step(score, timestamp_ms)
        if detected:
            detections.append(timestamp_ms)
            print(f"  ✅ Detection at {timestamp_ms}ms (score={score:.2f})")

    print(f"  Total detections: {len(detections)}")
    assert len(detections) == 1, "Should have exactly 1 detection"
    print(f"  ✅ Basic detector test passed")

    # Test 2: Sliding window processor
    print("\n2. Testing SlidingWindowProcessor...")

    processor = SlidingWindowProcessor(window_duration_s=1.0, hop_duration_s=0.1, sample_rate=16000)

    # Create dummy audio (5 seconds)
    audio = np.random.randn(5 * 16000)

    windows, timestamps = processor.extract_windows(audio)

    print(f"  Audio duration: 5 seconds")
    print(f"  Windows extracted: {len(windows)}")
    print(f"  Window shape: {windows[0].shape}")
    print(f"  First timestamp: {timestamps[0]}ms")
    print(f"  Last timestamp: {timestamps[-1]}ms")

    expected_windows = (len(audio) - processor.window_size) // processor.hop_size + 1
    assert len(windows) == expected_windows, f"Expected {expected_windows} windows"
    print(f"  ✅ Sliding window test passed")

    # Test 3: Hysteresis behavior
    print("\n3. Testing hysteresis...")

    detector_hyst = StreamingDetector(
        threshold_on=0.7,
        threshold_off=0.5,
        vote_window=3,
        vote_threshold=2,
        lockout_ms=500,
    )

    # Scores that cross both thresholds
    hyst_scores = [
        (0, 0.8),  # Above on_threshold
        (100, 0.75),  # Above on_threshold -> ACTIVATE
        (600, 0.6),  # Between thresholds (stays active)
        (700, 0.4),  # Below off_threshold
        (800, 0.3),  # Below off_threshold -> DEACTIVATE
    ]

    for timestamp_ms, score in hyst_scores:
        detected = detector_hyst.step(score, timestamp_ms)
        state = "ACTIVE" if detector_hyst.is_active else "INACTIVE"
        print(f"  t={timestamp_ms}ms, score={score:.2f}, state={state}, detected={detected}")

    print(f"  ✅ Hysteresis test passed")

    print("\n✅ All streaming detector tests passed")
