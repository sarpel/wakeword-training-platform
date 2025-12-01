"""
Real-Time Microphone Inference
Streaming audio capture and wakeword detection
"""
import queue
import threading
from typing import Callable, Optional, Tuple, Any

import numpy as np
import structlog
import torch
import torch.nn as nn

logger = structlog.get_logger(__name__)

try:
    import sounddevice as sd
except ImportError:
    sd = None
    logger.warning("sounddevice not installed. Microphone inference not available.")

from src.config.cuda_utils import enforce_cuda
from src.data.processor import AudioProcessor as GpuAudioProcessor
from pathlib import Path


class MicrophoneInference:
    """
    Real-time microphone inference for wakeword detection
    """

    def __init__(
        self,
        model: nn.Module,
        sample_rate: int = 16000,
        audio_duration: float = 1.5,
        threshold: float = 0.5,
        device: str = "cuda",
        callback: Optional[Callable[[float, bool], None]] = None,
        feature_type: str = "mel",
        n_mels: int = 128,
        n_mfcc: int = 40,
        n_fft: int = 1024,
        hop_length: int = 160,
    ):
        """
        Initialize microphone inference

        Args:
            model: Trained PyTorch model
            sample_rate: Audio sample rate
            audio_duration: Audio duration in seconds
            threshold: Detection threshold
            device: Device for inference
            callback: Callback function called on detection (confidence, is_positive)
            feature_type: Feature type ('mel' or 'mfcc')
            n_mels: Number of mel filterbanks
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        if sd is None:
            raise ImportError(
                "sounddevice not installed. " "Install with: pip install sounddevice"
            )

        # Enforce CUDA
        enforce_cuda()

        self.model = model
        self.sample_rate = sample_rate
        self.audio_duration = audio_duration
        self.chunk_samples = int(sample_rate * audio_duration)
        self.threshold = threshold
        self.device = device
        self.callback = callback

        # Move model to device and set eval mode
        self.model.to(device)
        self.model.eval()

        # Audio processor (GPU with CMVN)
        cmvn_path = Path("data/cmvn_stats.json")
        from src.config.defaults import WakewordConfig
        self.audio_processor = GpuAudioProcessor(
            config=WakewordConfig(), # Use defaults
            cmvn_path=cmvn_path if cmvn_path.exists() else None,
            device=device
        )
        
        # We don't need separate FeatureExtractor as GpuAudioProcessor handles it
        self.feature_extractor = None

        # Recording state
        self.is_recording = False
        self.audio_buffer = np.array([], dtype=np.float32)
        self.detection_count = 0
        self.false_alarm_count = 0

        # Thread-safe queues
        # Thread-safe queues
        self.audio_queue: queue.Queue = queue.Queue()
        self.result_queue: queue.Queue = queue.Queue()

        # Processing thread
        self.processing_thread: Optional[threading.Thread] = None

        logger.info("MicrophoneInference initialized")

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: dict, status: Any) -> None:
        """
        Callback for audio stream

        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Time information
            status: Status flags
        """
        if status:
            logger.warning(f"Audio callback status: {status}")

        # Add to queue for processing
        audio_chunk = indata[:, 0].copy()  # Get mono channel
        self.audio_queue.put(audio_chunk)

    def _processing_worker(self) -> None:
        """
        Background thread for audio processing
        """
        try:
            while self.is_recording:
                try:
                    # Get audio chunk (with timeout to allow checking is_recording)
                    audio_chunk = self.audio_queue.get(timeout=0.1)

                    # Add to buffer
                    self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])

                    # Process if we have enough samples
                    while len(self.audio_buffer) >= self.chunk_samples:
                        # Extract chunk
                        chunk = self.audio_buffer[: self.chunk_samples]
                        self.audio_buffer = self.audio_buffer[
                            self.chunk_samples // 2 :
                        ]  # 50% overlap

                        # Process chunk
                        confidence, is_positive = self._process_chunk(chunk)

                        # Put result in queue
                        self.result_queue.put((confidence, is_positive, chunk))

                        # Call callback if provided
                        if self.callback:
                            self.callback(confidence, is_positive)

                        # Update counts
                        if is_positive:
                            self.detection_count += 1
                        else:
                            self.false_alarm_count += 1

                except queue.Empty:
                    continue

        except Exception as e:
            logger.error(f"Processing worker error: {e}")
            logger.exception(e)

    def _process_chunk(self, audio_chunk: np.ndarray) -> Tuple[float, bool]:
        """
        Process audio chunk and return prediction

        Args:
            audio_chunk: Audio samples (1D array)

        Returns:
            Tuple of (confidence, is_positive)
        """
        try:
            # Normalize
            if np.max(np.abs(audio_chunk)) > 0:
                audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))

            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_chunk).float()

            # Extract features using GpuAudioProcessor (includes CMVN)
            # Input to GpuAudioProcessor should be (Batch, Samples)
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            audio_tensor = audio_tensor.to(self.device)
            features = self.audio_processor(audio_tensor)

            # Features are already (Batch, Channel, Freq, Time) or similar
            # No need to unsqueeze if processor returns 4D
            if features.ndim == 3:
                features = features.unsqueeze(1)

            # Inference
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    logits = self.model(features)
                # Convert to float32 immediately after inference to ensure compatibility
                logits = logits.float()

            # Get prediction
            probabilities = torch.softmax(logits, dim=1)
            confidence = probabilities[0, 1].item()
            is_positive = confidence >= self.threshold

            return confidence, is_positive

        except Exception as e:
            logger.error(f"Chunk processing error: {e}")
            return 0.0, False

    def start(self) -> None:
        """Start microphone recording and inference"""
        if self.is_recording:
            logger.warning("Already recording")
            return

        logger.info("Starting microphone inference...")

        # Reset state
        self.audio_buffer = np.array([], dtype=np.float32)
        self.detection_count = 0
        self.false_alarm_count = 0

        # Clear queues
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break

        # Start processing thread
        self.is_recording = True
        self.processing_thread = threading.Thread(
            target=self._processing_worker, daemon=True
        )
        self.processing_thread.start()

        # Start audio stream
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self._audio_callback,
            blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
        )
        self.stream.start()

        logger.info("Microphone inference started")

    def stop(self) -> None:
        """Stop microphone recording"""
        if not self.is_recording:
            logger.warning("Not recording")
            return

        logger.info("Stopping microphone inference...")

        # Stop recording
        self.is_recording = False

        # Stop stream
        if hasattr(self, "stream"):
            self.stream.stop()
            self.stream.close()

        # Wait for thread
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)

        logger.info(f"Microphone inference stopped. Detections: {self.detection_count}")

    def get_latest_result(self) -> Optional[Tuple[float, bool, np.ndarray]]:
        """
        Get latest detection result

        Returns:
            Tuple of (confidence, is_positive, audio_chunk) or None
        """
        try:
            from typing import cast
            return cast(Optional[Tuple[float, bool, np.ndarray]], self.result_queue.get_nowait())
        except queue.Empty:
            return None

    def get_stats(self) -> dict:
        """
        Get inference statistics

        Returns:
            Dictionary with statistics
        """
        return {
            "detection_count": self.detection_count,
            "false_alarm_count": self.false_alarm_count,
            "is_recording": self.is_recording,
            "buffer_size": len(self.audio_buffer),
        }


class SimulatedMicrophoneInference:
    """
    Simulated microphone inference for testing without actual microphone
    """

    def __init__(
        self,
        model: nn.Module,
        sample_rate: int = 16000,
        audio_duration: float = 1.5,
        threshold: float = 0.5,
        device: str = "cuda",
        callback: Optional[Callable] = None,
    ) -> None:
        """Initialize simulated microphone inference"""
        self.model = model
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.callback = callback
        self.is_recording = False
        self.detection_count = 0

        logger.info("SimulatedMicrophoneInference initialized (no real audio)")

    def start(self) -> None:
        """Start simulated recording"""
        self.is_recording = True
        self.detection_count = 0
        logger.info("Simulated microphone started")

    def stop(self) -> None:
        """Stop simulated recording"""
        self.is_recording = False
        logger.info("Simulated microphone stopped")

    def get_latest_result(self) -> Optional[Tuple[float, bool, np.ndarray]]:
        """Get simulated result"""
        if self.is_recording:
            # Return random results
            import random

            confidence = random.random()
            is_positive = confidence >= self.threshold
            dummy_audio = np.random.randn(int(self.sample_rate * 1.5)).astype(
                np.float32
            )
            return confidence, is_positive, dummy_audio
        return None

    def get_stats(self) -> dict:
        """Get simulated stats"""
        return {
            "detection_count": self.detection_count,
            "false_alarm_count": 0,
            "is_recording": self.is_recording,
            "buffer_size": 0,
        }


if __name__ == "__main__":
    # Test microphone inference
    print("Microphone Inference Test")
    print("=" * 60)

    if sd is None:
        print("⚠️  sounddevice not installed")
        print("Install with: pip install sounddevice")
    else:
        print(f"✅ sounddevice available")

        # List audio devices
        print("\nAvailable audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                print(f"  {i}: {device['name']}")

    print("\nMicrophone inference module ready")
