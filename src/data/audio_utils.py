"""
Audio Utilities for Wakeword Training Platform
File validation, loading, and basic processing
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import librosa
import numpy as np
import soundfile as sf
import structlog

from src.exceptions import AudioProcessingError, DataLoadError

logger = structlog.get_logger(__name__)


class AudioValidator:
    """Validates audio files and extracts metadata"""

    SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    def __init__(self) -> None:
        self.valid_files: List[Path] = []
        self.corrupted_files: List[Path] = []
        self.unsupported_files: List[Path] = []

    @staticmethod
    def is_audio_file(file_path: Path) -> bool:
        """
        Check if file is a supported audio format

        Args:
            file_path: Path to file

        Returns:
            True if supported audio format
        """
        return file_path.suffix.lower() in AudioValidator.SUPPORTED_FORMATS

    @staticmethod
    def validate_audio_file(
        file_path: Path,
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Validate audio file and extract metadata

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (is_valid, metadata_dict, error_message)
        """
        try:
            # Try to read audio file
            info = sf.info(str(file_path))

            metadata: Dict[str, Any] = {
                "path": str(file_path),
                "filename": file_path.name,
                "format": file_path.suffix.lower(),
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "duration": info.duration,
                "frames": info.frames,
                "subtype": info.subtype,
            }

            return True, metadata, None

        except Exception as e:
            error_msg = f"Error validating {file_path.name}: {str(e)}"
            logger.debug(error_msg)
            raise DataLoadError(error_msg) from e

    @staticmethod
    def load_audio(
        file_path: Path,
        target_sr: int = 16000,
        mono: bool = True,
        duration: Optional[float] = None,
        offset: float = 0.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file with resampling and mono conversion

        Args:
            file_path: Path to audio file
            target_sr: Target sample rate
            mono: Convert to mono
            duration: Duration to load (None = all)
            offset: Start time in seconds

        Returns:
            Tuple of (audio_array, sample_rate)
            Note: librosa.load returns float for sr, but we cast to int for consistency
        """
        try:
            # Load with librosa (handles various formats and resampling)
            audio, sr = librosa.load(
                str(file_path),
                sr=target_sr,
                mono=mono,
                duration=duration,
                offset=offset,
            )

            # Cast sample rate to int for type consistency
            return audio, int(sr)

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise AudioProcessingError(f"Failed to load audio file: {file_path}") from e

    @staticmethod
    def get_audio_stats(audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Calculate audio statistics

        Args:
            audio: Audio array
            sr: Sample rate

        Returns:
            Dictionary of statistics
        """
        return {
            "duration": len(audio) / sr,
            "samples": len(audio),
            "sample_rate": sr,
            "rms": float(np.sqrt(np.mean(audio**2))),
            "max_amplitude": float(np.max(np.abs(audio))),
            "min_value": float(np.min(audio)),
            "max_value": float(np.max(audio)),
            "mean": float(np.mean(audio)),
            "std": float(np.std(audio)),
            "zero_crossing_rate": float(np.mean(librosa.zero_crossings(audio))),
        }

    @staticmethod
    def check_audio_quality(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check audio quality and return warnings/exclusion flags

        Args:
            metadata: Audio metadata

        Returns:
            Dictionary with quality check results and exclusion status
        """
        warnings = []
        quality_score = 100.0
        should_exclude = False
        exclude_reason = None

        # Determine context from path
        is_rir = "rirs" in str(metadata.get("path", "")).lower()

        # Check sample rate
        if metadata["sample_rate"] < 4000:
            should_exclude = True
            exclude_reason = f"Extremely low sample rate: {metadata['sample_rate']}Hz"
            quality_score = 0
        elif metadata["sample_rate"] < 8000:
            warnings.append(f"Very low sample rate: {metadata['sample_rate']}Hz")
            quality_score -= 30
        elif metadata["sample_rate"] < 16000:
            warnings.append(
                f"Low sample rate: {metadata['sample_rate']}Hz (16kHz recommended)"
            )
            quality_score -= 10

        # Check duration
        # RIRs are naturally short (impulses), so we allow down to 0.1s
        min_duration = 0.1 if is_rir else 0.4
        
        if metadata["duration"] < min_duration:
            should_exclude = True
            exclude_reason = f"Too short: {metadata['duration']:.2f}s (min {min_duration}s)"
            quality_score = 0
        elif metadata["duration"] > 2.5:
            should_exclude = True
            exclude_reason = f"Too long: {metadata['duration']:.2f}s (max 2.5s for ESP32)"
            quality_score = 0
        elif metadata["duration"] < 0.3 and not is_rir:
            warnings.append(
                f"Short duration: {metadata['duration']:.2f}s (0.6s+ recommended)"
            )
            quality_score -= 10
        elif metadata["duration"] > 1.5:
            warnings.append(
                f"Long duration: {metadata['duration']:.2f}s (1.5s recommended for 'Hey Katya')"
            )
            quality_score -= 5

        # Check channels (Informational only, not a quality penalty)
        if metadata["channels"] > 1:
            # Stereo is handled automatically, so we don't penalize score, just note it if needed
            pass

        return {
            "quality_score": max(0, quality_score),
            "warnings": warnings,
            "is_acceptable": quality_score >= 50 and not should_exclude,
            "should_exclude": should_exclude,
            "exclude_reason": exclude_reason,
        }


class AudioProcessor:
    """Process audio files for training"""

    def __init__(self, target_sr: int = 16000, target_duration: float = 1.5):
        """
        Initialize audio processor

        Args:
            target_sr: Target sample rate
            target_duration: Target duration in seconds
        """
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_samples = int(target_sr * target_duration)

    def process_audio(self, file_path: Path, pad_mode: str = "constant") -> np.ndarray:
        """
        Load and process audio file to standard format

        Args:
            file_path: Path to audio file
            pad_mode: Padding mode ('constant', 'edge', 'wrap')

        Returns:
            Processed audio array (mono, target_sr, target_duration)
        """
        # Load audio
        audio, sr = AudioValidator.load_audio(
            file_path, target_sr=self.target_sr, mono=True
        )

        # Normalize length
        audio = self._normalize_length(audio, pad_mode)

        # Normalize amplitude
        audio = self._normalize_amplitude(audio)

        return audio

    def _normalize_length(self, audio: np.ndarray, pad_mode: str) -> np.ndarray:
        """
        Normalize audio to target length

        Args:
            audio: Input audio
            pad_mode: Padding mode

        Returns:
            Audio with target length
        """
        current_samples = len(audio)

        if current_samples == self.target_samples:
            return audio
        elif current_samples > self.target_samples:
            # Trim from center
            start = (current_samples - self.target_samples) // 2
            return audio[start : start + self.target_samples]
        else:
            # Pad
            pad_amount = self.target_samples - current_samples
            pad_left = pad_amount // 2
            pad_right = pad_amount - pad_left

            if pad_mode == "constant":
                return np.pad(audio, (pad_left, pad_right), mode="constant")
            else:
                # Mypy: np.pad mode parameter needs explicit type
                # Cast pad_mode to satisfy numpy's type requirements
                return np.pad(audio, (pad_left, pad_right), mode=pad_mode)  # type: ignore[no-any-return, arg-type, call-overload]

    def _normalize_amplitude(
        self, audio: np.ndarray, target_level: float = 0.3
    ) -> np.ndarray:
        """
        Normalize audio amplitude

        Args:
            audio: Input audio
            target_level: Target RMS level

        Returns:
            Normalized audio
        """
        # BUG FIX: Added check for empty/zero audio to prevent division by zero
        # Also added epsilon to RMS calculation for numerical stability
        if len(audio) == 0:
            # Empty audio - return as is
            return audio

        rms = np.sqrt(np.mean(audio**2))

        # BUG FIX: Added epsilon (1e-8) to prevent division by zero for silent audio
        if rms > 1e-8:
            audio = audio * (target_level / rms)

        # Clip to prevent overflow
        audio = np.clip(audio, -1.0, 1.0)

        return audio

    @staticmethod
    def trim_silence(
        audio: np.ndarray, top_db: float = 20.0, frame_length: int = 2048, hop_length: int = 512
    ) -> np.ndarray:
        """
        Trim leading and trailing silence from audio

        Args:
            audio: Input audio array
            top_db: The threshold (in decibels) below reference to consider as silence
            frame_length: The number of samples per analysis frame
            hop_length: The number of samples between analysis frames

        Returns:
            Trimmed audio array
        """
        # librosa.effects.trim returns (trimmed_audio, index)
        trimmed_audio, _ = librosa.effects.trim(
            audio, top_db=top_db, frame_length=frame_length, hop_length=hop_length
        )
        return trimmed_audio


def scan_audio_files(directory: Path, recursive: bool = True) -> List[Path]:
    """
    Scan directory for audio files

    Args:
        directory: Directory to scan
        recursive: Search recursively in subdirectories

    Returns:
        List of audio file paths
    """
    audio_files: List[Path] = []

    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return audio_files

    if recursive:
        # Recursively find all audio files
        for ext in AudioValidator.SUPPORTED_FORMATS:
            audio_files.extend(directory.rglob(f"*{ext}"))
            audio_files.extend(directory.rglob(f"*{ext.upper()}"))
    else:
        # Only immediate directory
        for ext in AudioValidator.SUPPORTED_FORMATS:
            audio_files.extend(directory.glob(f"*{ext}"))
            audio_files.extend(directory.glob(f"*{ext.upper()}"))

    return sorted(set(audio_files))  # Remove duplicates and sort


if __name__ == "__main__":
    # Test audio utilities
    pass

    from src.config.logger import get_logger

    test_logger = get_logger("audio_utils_test")

    # Test validation
    validator = AudioValidator()
    test_logger.info("AudioValidator created successfully")

    # Test processor
    processor = AudioProcessor(target_sr=16000, target_duration=1.5)
    test_logger.info("AudioProcessor created successfully")

    test_logger.info("Audio utilities test complete")
