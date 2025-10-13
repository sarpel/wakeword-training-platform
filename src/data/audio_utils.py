"""
Audio Utilities for Wakeword Training Platform
File validation, loading, and basic processing
"""
import soundfile as sf
import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class AudioValidator:
    """Validates audio files and extracts metadata"""

    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

    def __init__(self):
        self.valid_files = []
        self.corrupted_files = []
        self.unsupported_files = []

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
    def validate_audio_file(file_path: Path) -> Tuple[bool, Optional[Dict], Optional[str]]:
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

            metadata = {
                'path': str(file_path),
                'filename': file_path.name,
                'format': file_path.suffix.lower(),
                'sample_rate': info.samplerate,
                'channels': info.channels,
                'duration': info.duration,
                'frames': info.frames,
                'subtype': info.subtype,
            }

            return True, metadata, None

        except Exception as e:
            error_msg = f"Error validating {file_path.name}: {str(e)}"
            logger.debug(error_msg)
            return False, None, error_msg

    @staticmethod
    def load_audio(
        file_path: Path,
        target_sr: int = 16000,
        mono: bool = True,
        duration: Optional[float] = None,
        offset: float = 0.0
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
        """
        try:
            # Load with librosa (handles various formats and resampling)
            audio, sr = librosa.load(
                str(file_path),
                sr=target_sr,
                mono=mono,
                duration=duration,
                offset=offset
            )

            return audio, sr

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise

    @staticmethod
    def get_audio_stats(audio: np.ndarray, sr: int) -> Dict:
        """
        Calculate audio statistics

        Args:
            audio: Audio array
            sr: Sample rate

        Returns:
            Dictionary of statistics
        """
        return {
            'duration': len(audio) / sr,
            'samples': len(audio),
            'sample_rate': sr,
            'rms': float(np.sqrt(np.mean(audio**2))),
            'max_amplitude': float(np.max(np.abs(audio))),
            'min_value': float(np.min(audio)),
            'max_value': float(np.max(audio)),
            'mean': float(np.mean(audio)),
            'std': float(np.std(audio)),
            'zero_crossing_rate': float(np.mean(librosa.zero_crossings(audio))),
        }

    @staticmethod
    def check_audio_quality(metadata: Dict) -> Dict[str, Any]:
        """
        Check audio quality and return warnings

        Args:
            metadata: Audio metadata

        Returns:
            Dictionary with quality check results
        """
        warnings = []
        quality_score = 100.0

        # Check sample rate
        if metadata['sample_rate'] < 8000:
            warnings.append(f"Very low sample rate: {metadata['sample_rate']}Hz")
            quality_score -= 30
        elif metadata['sample_rate'] < 16000:
            warnings.append(f"Low sample rate: {metadata['sample_rate']}Hz (16kHz recommended)")
            quality_score -= 10

        # Check duration
        if metadata['duration'] < 0.4:
            warnings.append(f"Very short duration: {metadata['duration']:.2f}s")
            quality_score -= 20
        elif metadata['duration'] < 0.5:
            warnings.append(f"Short duration: {metadata['duration']:.2f}s (1-2s typical)")
            quality_score -= 5
        elif metadata['duration'] > 4.0:
            warnings.append(f"Long duration: {metadata['duration']:.2f}s (may need trimming)")
            quality_score -= 5

        # Check channels
        if metadata['channels'] > 1:
            warnings.append(f"Stereo audio detected ({metadata['channels']} channels), will convert to mono")
            quality_score -= 5

        return {
            'quality_score': max(0, quality_score),
            'warnings': warnings,
            'is_acceptable': quality_score >= 50
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

    def process_audio(
        self,
        file_path: Path,
        pad_mode: str = 'constant'
    ) -> np.ndarray:
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
            file_path,
            target_sr=self.target_sr,
            mono=True
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
            return audio[start:start + self.target_samples]
        else:
            # Pad
            pad_amount = self.target_samples - current_samples
            pad_left = pad_amount // 2
            pad_right = pad_amount - pad_left

            if pad_mode == 'constant':
                return np.pad(audio, (pad_left, pad_right), mode='constant')
            else:
                return np.pad(audio, (pad_left, pad_right), mode=pad_mode)

    def _normalize_amplitude(self, audio: np.ndarray, target_level: float = 0.3) -> np.ndarray:
        """
        Normalize audio amplitude

        Args:
            audio: Input audio
            target_level: Target RMS level

        Returns:
            Normalized audio
        """
        rms = np.sqrt(np.mean(audio**2))

        if rms > 0:
            audio = audio * (target_level / rms)

        # Clip to prevent overflow
        audio = np.clip(audio, -1.0, 1.0)

        return audio


def scan_audio_files(directory: Path, recursive: bool = True) -> list:
    """
    Scan directory for audio files

    Args:
        directory: Directory to scan
        recursive: Search recursively in subdirectories

    Returns:
        List of audio file paths
    """
    audio_files = []

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
    import sys
    from src.config.logger import get_logger

    test_logger = get_logger("audio_utils_test")

    # Test validation
    validator = AudioValidator()
    test_logger.info("AudioValidator created successfully")

    # Test processor
    processor = AudioProcessor(target_sr=16000, target_duration=1.5)
    test_logger.info("AudioProcessor created successfully")

    test_logger.info("Audio utilities test complete")