"""
Audio Utilities for Wakeword Training Platform
File validation, loading, and basic processing
"""

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
            warnings.append(f"Low sample rate: {metadata['sample_rate']}Hz (16kHz recommended)")
            quality_score -= 10

        # Check duration
        # RIRs are naturally short (impulses), so they have separate duration limits
        if is_rir:
            min_duration = 0.05  # Very short impulse
            max_duration = 3.0   # Allow longer RIRs
        else:
            min_duration = 0.4
            max_duration = 2.5

        if metadata["duration"] < min_duration:
            should_exclude = True
            exclude_reason = f"Too short: {metadata['duration']:.2f}s (min {min_duration}s)"
            quality_score = 0
        elif metadata["duration"] > max_duration:
            should_exclude = True
            exclude_reason = f"Too long: {metadata['duration']:.2f}s (max {max_duration}s)"
            quality_score = 0
        elif not is_rir:
            # Quality warnings for non-RIR files
            if metadata["duration"] < 0.4:
                warnings.append(f"Short duration: {metadata['duration']:.2f}s (0.4s+ recommended)")
                quality_score -= 10
            elif metadata["duration"] > 2.5:
                warnings.append(f"Long duration: {metadata['duration']:.2f}s (2.5s max for ESP32)")
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
        audio, sr = AudioValidator.load_audio(file_path, target_sr=self.target_sr, mono=True)

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

    def _normalize_amplitude(self, audio: np.ndarray, target_level: float = 0.3) -> np.ndarray:
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
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
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


class SilentDetector:
    """Detects and moves silent or near-silent audio files"""
    
    def __init__(self, rms_threshold: float = -50.0, peak_threshold: Optional[float] = None, 
                 min_duration: float = 0.25) -> None:
        """
        Initialize silent detector
        
        Args:
            rms_threshold: RMS threshold in dB (default: -50dB)
            peak_threshold: Peak threshold in dB (optional)
            min_duration: Files shorter than this are not considered silent (default: 0.25s)
        """
        self.rms_threshold = rms_threshold
        self.peak_threshold = peak_threshold
        self.min_duration = min_duration
        self.silent_files: List[Dict[str, Any]] = []
        
    def _to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert to mono if needed"""
        if audio.ndim == 1:
            return audio.astype(np.float32, copy=False)
        return audio.mean(axis=1).astype(np.float32, copy=False)
    
    def _dbfs_from_rms(self, audio: np.ndarray, eps: float = 1e-12) -> float:
        """Calculate dBFS from RMS"""
        rms = float(np.sqrt(np.mean(audio * audio)) + eps)
        return 20.0 * np.log10(rms)
    
    def _dbfs_from_peak(self, audio: np.ndarray, eps: float = 1e-12) -> float:
        """Calculate dBFS from peak"""
        peak = float(np.max(np.abs(audio)) + eps)
        return 20.0 * np.log10(peak)
    
    def is_silent(self, audio: np.ndarray, sr: int, duration: Optional[float] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if audio is silent or near-silent
        
        Args:
            audio: Audio array
            sr: Sample rate
            duration: Override duration (calculated from audio if None)
            
        Returns:
            Tuple of (is_silent, info_dict)
        """
        if duration is None:
            duration = len(audio) / sr
            
        # Skip files shorter than minimum duration
        if duration < self.min_duration:
            return False, {"reason": f"duration<{self.min_duration}s"}
        
        # Convert to mono
        audio_mono = self._to_mono(audio)
        
        # Check for empty audio
        if audio_mono.size == 0:
            return True, {"reason": "empty"}
            
        # Calculate RMS and Peak dBFS
        rms_dbfs = self._dbfs_from_rms(audio_mono)
        peak_dbfs = self._dbfs_from_peak(audio_mono)
        
        info = {
            "duration_sec": duration,
            "rms_dbfs": rms_dbfs,
            "peak_dbfs": peak_dbfs
        }
        
        # Check RMS threshold
        is_silent = rms_dbfs <= self.rms_threshold
        
        # Check peak threshold if specified
        if self.peak_threshold is not None:
            is_silent = is_silent and (peak_dbfs <= self.peak_threshold)
            
        return is_silent, info
        
    def analyze_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Analyze a single audio file for silence
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with results or None if file cannot be processed
        """
        try:
            # Load audio
            audio, sr = AudioValidator.load_audio(file_path, mono=True)
            duration = len(audio) / sr
            is_silent, info = self.is_silent(audio, sr, duration)
            
            result = {
                "path": str(file_path),
                "filename": file_path.name,
                "is_silent": is_silent,
                "duration": duration,
                **info  # Include rms_dbfs, peak_dbfs, reason, etc.
            }
            
            # For backward compatibility
            if "rms_dbfs" in info:
                result["rms_db"] = float(info["rms_dbfs"])
                
            return result
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return None
            
    def analyze_directory(self, directory: Path, recursive: bool = True) -> List[Dict[str, Any]]:
        """
        Analyze all audio files in directory for silence
        
        Args:
            directory: Directory to analyze
            recursive: Search recursively
            
        Returns:
            List of silent file analyses
        """
        audio_files = scan_audio_files(directory, recursive)
        silent_files = []
        
        logger.info(f"Analyzing {len(audio_files)} files for silence...")
        
        for file_path in audio_files:
            result = self.analyze_file(file_path)
            if result and result["is_silent"]:
                silent_files.append(result)
                
        self.silent_files = silent_files
        logger.info(f"Found {len(silent_files)} silent files")
        
        return silent_files
        
    def move_silent_files(self, silent_root: Path = Path("silent_dataset"), 
                         source_dirs: Optional[List[Path]] = None,
                         copy_mode: bool = False, dry_run: bool = False,
                         keep_structure: bool = True) -> Tuple[int, Dict[str, Any]]:
        """
        Move or copy silent files to a separate directory
        
        Args:
            silent_root: Directory to move silent files to
            source_dirs: List of source directories to preserve structure
            copy_mode: Copy instead of move (default: False)
            dry_run: Only report, don't actually move files (default: False)
            keep_structure: Preserve directory structure (default: True)
            
        Returns:
            Tuple of (files_processed, report_data)
        """
        if not self.silent_files:
            logger.warning("No silent files found. Run analyze_directory() first.")
            return 0, {}
            
        silent_root = Path(silent_root)
        if not dry_run:
            silent_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"{'Copying' if copy_mode else 'Moving'} silent files to: {silent_root}")
        
        moved_count = 0
        report_data = []
        
        for file_info in self.silent_files:
            src_path = Path(file_info["path"])
            
            if not src_path.exists():
                logger.warning(f"File not found: {src_path}")
                continue
                
            # Create destination path
            if keep_structure and source_dirs:
                for source_dir in source_dirs:
                    source_dir_path = Path(source_dir)
                    try:
                        rel_path = src_path.relative_to(source_dir_path)
                        dest_dir = silent_root / rel_path.parent
                        break
                    except ValueError:
                        continue
                else:
                    category = src_path.parent.name
                    dest_dir = silent_root / category
            else:
                category = src_path.parent.name
                dest_dir = silent_root / category
                
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / src_path.name
            
            # Handle duplicates
            if dest_path.exists():
                stem, suf = src_path.stem, src_path.suffix
                i = 1
                while True:
                    cand = dest_dir / f"{stem}_{i}{suf}"
                    if not cand.exists():
                        dest_path = cand
                        break
                    i += 1
            
            # Process file
            try:
                if not dry_run:
                    if copy_mode:
                        shutil.copy2(str(src_path), str(dest_path))
                    else:
                        shutil.move(str(src_path), str(dest_path))
                
                moved_count += 1
                
                # Add to report
                report_entry = {
                    "status": "processed",
                    "path": str(src_path),
                    "duration_sec": file_info.get("duration", ""),
                    "rms_dbfs": file_info.get("rms_dbfs", ""),
                    "peak_dbfs": file_info.get("peak_dbfs", ""),
                    "reason": file_info.get("reason", ""),
                    "target": str(dest_path)
                }
                report_data.append(report_entry)
                
                logger.info(f"{'Copied' if copy_mode else 'Moved'} silent file: {src_path.name} (RMS: {file_info.get('rms_dbfs', '?'):.1f}dB) -> {dest_path}")
                
            except Exception as e:
                logger.error(f"Failed to process {src_path}: {e}")
        
        logger.info(f"Processed {moved_count} silent files to {silent_root} (dry_run={dry_run})")
        return moved_count, {"silent_files": report_data}

def move_silent_files_from_directory(
    directory: Path, 
    silent_root: Path = Path("silent_dataset"), 
    rms_threshold: float = -50.0,
    peak_threshold: Optional[float] = None,
    min_duration: float = 0.25,
    recursive: bool = True,
    copy_mode: bool = False,
    dry_run: bool = False,
    keep_structure: bool = True,
    save_report: bool = True,
    report_name: str = "silent_report.tsv"
) -> Tuple[int, Dict[str, Any]]:
    """
    Utility function to detect and move silent files from a directory
    
    Args:
        directory: Directory to scan for silent files
        silent_root: Directory to move silent files to
        rms_threshold: RMS threshold in dB (default: -50dB)
        peak_threshold: Peak threshold in dB (optional)
        min_duration: Files shorter than this are not considered silent (default: 0.25s)
        recursive: Search recursively in subdirectories
        copy_mode: Copy instead of move (default: False)
        dry_run: Only report, don't actually move files (default: False)
        keep_structure: Preserve directory structure (default: True)
        save_report: Save TSV report file (default: True)
        report_name: Report file name (default: "silent_report.tsv")
        
    Returns:
        Tuple of (files_processed, report_data)
    """
    detector = SilentDetector(
        rms_threshold=rms_threshold,
        peak_threshold=peak_threshold,
        min_duration=min_duration
    )
    
    silent_files = detector.analyze_directory(directory, recursive)
    
    if not silent_files:
        logger.info("No silent files found")
        return 0, {"silent_files": [], "total_scanned": 0}
    
    moved_count, report_data = detector.move_silent_files(
        silent_root, 
        source_dirs=[directory],
        copy_mode=copy_mode,
        dry_run=dry_run,
        keep_structure=keep_structure
    )
    
    # Save report if requested
    if save_report and not dry_run:
        report_path = Path(silent_root) / report_name
        with report_path.open("w", encoding="utf-8") as f:
            f.write("status\tpath\tduration_sec\trms_dbfs\tpeak_dbfs\treason\ttarget\n")
            
            for entry in report_data.get("silent_files", []):
                f.write(f"silent\t{entry['path']}\t{entry['duration_sec']}\t"
                       f"{entry['rms_dbfs']}\t{entry['peak_dbfs']}\t"
                       f"{entry['reason']}\t{entry['target']}\n")
        
        logger.info(f"Report saved to: {report_path}")
    
    total_scanned = len(scan_audio_files(directory, recursive))
    logger.info(f"Silent files processed: {moved_count}/{total_scanned} (dry_run={dry_run})")
    
    return moved_count, {
        **report_data,
        "total_scanned": total_scanned,
        "report_path": str(Path(silent_root) / report_name) if save_report else None
    }


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
