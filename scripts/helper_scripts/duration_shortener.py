"""
Audio Duration Shortener
Splits long audio files into optimal duration chunks for wakeword training

Industry standard for background noise: 3-5 seconds
This script splits audio files into 4-second chunks with 0.5s overlap
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

# Audio processing
try:
    import librosa
    import numpy as np
    import soundfile as sf
except ImportError:
    print("ERROR: Required libraries not found!")
    print("Please install: pip install librosa soundfile numpy")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AudioDurationShortener:
    """
    Shortens audio files to optimal duration for wakeword training

    Industry standard: 3-5 seconds for background noise
    Default: 4 seconds with 0.5s overlap for smooth transitions
    """

    def __init__(
        self,
        target_duration: float = 4.0,
        overlap: float = 0.5,
        min_duration: float = 2.0,
        sample_rate: int = 16000,
        backup: bool = True,
    ):
        """
        Initialize shortener

        Args:
            target_duration: Target duration in seconds (default: 4.0)
            overlap: Overlap between chunks in seconds (default: 0.5)
            min_duration: Minimum duration for last chunk (default: 2.0)
            sample_rate: Target sample rate (default: 16000)
            backup: Create backup before processing (default: True)
        """
        self.target_duration = target_duration
        self.overlap = overlap
        self.min_duration = min_duration
        self.sample_rate = sample_rate
        self.backup = backup

        logger.info(f"AudioDurationShortener initialized:")
        logger.info(f"  Target duration: {target_duration}s")
        logger.info(f"  Overlap: {overlap}s")
        logger.info(f"  Min duration: {min_duration}s")
        logger.info(f"  Sample rate: {sample_rate}Hz")
        logger.info(f"  Backup: {backup}")

    def process_directory(self, directory: Path, recursive: bool = True) -> Tuple[int, int]:
        """
        Process all audio files in directory

        Args:
            directory: Directory path
            recursive: Process subdirectories (default: True)

        Returns:
            Tuple of (files_processed, chunks_created)
        """
        directory = Path(directory)

        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return 0, 0

        # Find audio files
        extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
        audio_files: List[Path] = []

        if recursive:
            for ext in extensions:
                audio_files.extend(directory.rglob(f"*{ext}"))
        else:
            for ext in extensions:
                audio_files.extend(directory.glob(f"*{ext}"))

        logger.info(f"Found {len(audio_files)} audio files in {directory}")

        # Create backup if enabled
        if self.backup and audio_files:
            backup_dir = directory / "backup_original"
            if not backup_dir.exists():
                backup_dir.mkdir(parents=True)
                logger.info(f"Created backup directory: {backup_dir}")

        files_processed = 0
        total_chunks = 0

        for audio_file in audio_files:
            try:
                # Skip if already in backup directory
                if "backup_original" in str(audio_file):
                    continue

                chunks = self.process_file(audio_file)

                if chunks > 0:
                    files_processed += 1
                    total_chunks += chunks
                    logger.info(f"✅ Processed: {audio_file.name} → {chunks} chunks")
                else:
                    logger.info(f"⏭️  Skipped: {audio_file.name} (already optimal duration)")

            except Exception as e:
                logger.error(f"❌ Error processing {audio_file.name}: {e}")

        logger.info("=" * 80)
        logger.info(f"Processing complete!")
        logger.info(f"  Files processed: {files_processed}")
        logger.info(f"  Total chunks created: {total_chunks}")
        logger.info("=" * 80)

        return files_processed, total_chunks

    def process_file(self, file_path: Path) -> int:
        """
        Process single audio file

        Args:
            file_path: Path to audio file

        Returns:
            Number of chunks created (0 if file already optimal)
        """
        file_path = Path(file_path)

        # Load audio
        audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        duration = len(audio) / sr

        # Check if file needs processing
        if duration <= self.target_duration:
            logger.debug(f"File {file_path.name} is already optimal ({duration:.2f}s)")
            return 0

        # Create backup
        if self.backup:
            backup_dir = file_path.parent / "backup_original"
            backup_dir.mkdir(exist_ok=True)
            backup_path = backup_dir / file_path.name

            if not backup_path.exists():
                shutil.copy2(file_path, backup_path)
                logger.debug(f"Backed up: {file_path.name}")

        # Split into chunks
        chunks = self._split_audio(audio, int(sr))

        # Save chunks
        base_name = file_path.stem
        extension = file_path.suffix
        parent_dir = file_path.parent

        for i, chunk in enumerate(chunks):
            chunk_name = f"{base_name}_part{i+1:03d}{extension}"
            chunk_path = parent_dir / chunk_name

            # Save chunk
            sf.write(chunk_path, chunk, sr)
            logger.debug(f"Created chunk: {chunk_name}")

        # Remove original file
        file_path.unlink()
        logger.debug(f"Removed original: {file_path.name}")

        return len(chunks)

    def _split_audio(self, audio: np.ndarray, sr: int) -> List[np.ndarray]:
        """
        Split audio into overlapping chunks

        Args:
            audio: Audio array
            sr: Sample rate

        Returns:
            List of audio chunks
        """
        duration = len(audio) / sr

        # Calculate chunk parameters
        chunk_samples = int(self.target_duration * sr)
        overlap_samples = int(self.overlap * sr)
        hop_samples = chunk_samples - overlap_samples

        chunks = []
        start = 0

        while start < len(audio):
            end = start + chunk_samples

            # Extract chunk
            chunk = audio[start:end]

            # Check if last chunk is too short
            chunk_duration = len(chunk) / sr

            if chunk_duration >= self.min_duration:
                # Pad last chunk if needed
                if len(chunk) < chunk_samples:
                    padding = chunk_samples - len(chunk)
                    chunk = np.pad(chunk, (0, padding), mode="constant")

                chunks.append(chunk)
            else:
                # Last chunk too short, merge with previous
                if chunks:
                    chunks[-1] = np.concatenate([chunks[-1][:hop_samples], chunk])
                else:
                    # Single chunk case
                    chunks.append(chunk)

            start += hop_samples

        return chunks

    def get_statistics(self, directory: Path, recursive: bool = True) -> Dict[str, Any]:
        """
        Get statistics about audio files in directory

        Args:
            directory: Directory path
            recursive: Check subdirectories

        Returns:
            Dictionary with statistics
        """
        directory = Path(directory)
        extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
        audio_files: List[Path] = []

        if recursive:
            for ext in extensions:
                audio_files.extend(directory.rglob(f"*{ext}"))
        else:
            for ext in extensions:
                audio_files.extend(directory.glob(f"*{ext}"))

        durations = []
        total_size = 0

        for audio_file in audio_files:
            # Skip backup files
            if "backup_original" in str(audio_file):
                continue

            try:
                audio, sr = librosa.load(audio_file, sr=None, duration=1)
                info = sf.info(audio_file)
                duration = info.duration
                durations.append(duration)
                total_size += audio_file.stat().st_size
            except Exception as e:
                logger.warning(f"Could not read {audio_file.name}: {e}")

        if not durations:
            return {"total_files": 0, "avg_duration": 0, "min_duration": 0, "max_duration": 0, "total_size_mb": 0}

        stats = {
            "total_files": len(durations),
            "avg_duration": np.mean(durations),
            "min_duration": np.min(durations),
            "max_duration": np.max(durations),
            "std_duration": np.std(durations),
            "total_size_mb": total_size / (1024 * 1024),
            "files_needing_split": sum(1 for d in durations if d > self.target_duration),
        }

        return stats


def main() -> None:
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Split long audio files into optimal duration chunks for wakeword training"
    )
    parser.add_argument("directory", type=str, help="Directory containing audio files")
    parser.add_argument("--duration", type=float, default=4.0, help="Target duration in seconds (default: 4.0)")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap between chunks in seconds (default: 0.5)")
    parser.add_argument(
        "--min-duration", type=float, default=2.0, help="Minimum duration for last chunk (default: 2.0)"
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate (default: 16000)")
    parser.add_argument("--no-backup", action="store_true", help="Do not create backup of original files")
    parser.add_argument("--no-recursive", action="store_true", help="Do not process subdirectories")
    parser.add_argument("--stats-only", action="store_true", help="Only show statistics, do not process")

    args = parser.parse_args()

    # Create shortener
    shortener = AudioDurationShortener(
        target_duration=args.duration,
        overlap=args.overlap,
        min_duration=args.min_duration,
        sample_rate=args.sample_rate,
        backup=not args.no_backup,
    )

    directory = Path(args.directory)

    # Show statistics
    print("\n" + "=" * 80)
    print("AUDIO DURATION ANALYSIS")
    print("=" * 80)

    stats = shortener.get_statistics(directory, recursive=not args.no_recursive)

    print(f"\nDirectory: {directory}")
    print(f"Total files: {stats['total_files']}")
    print(f"Average duration: {stats['avg_duration']:.2f}s")
    print(f"Min duration: {stats['min_duration']:.2f}s")
    print(f"Max duration: {stats['max_duration']:.2f}s")
    print(f"Std duration: {stats.get('std_duration', 0):.2f}s")
    print(f"Total size: {stats['total_size_mb']:.2f} MB")
    print(f"\nFiles needing split (>{args.duration}s): {stats['files_needing_split']}")

    # Estimate chunks
    if stats["files_needing_split"] > 0:
        avg_long_duration = (
            np.mean([d for d in [stats["avg_duration"]] if d > args.duration])
            if stats["avg_duration"] > args.duration
            else stats["max_duration"]
        )

        estimated_chunks = int(avg_long_duration / (args.duration - args.overlap)) * stats["files_needing_split"]
        print(f"Estimated chunks to create: ~{estimated_chunks}")

    if args.stats_only:
        print("\n(Statistics only - no files processed)")
        return

    # Confirm processing
    if stats["files_needing_split"] > 0:
        print("\n" + "=" * 80)
        print("⚠️  WARNING: This will modify audio files!")
        print("=" * 80)
        print(f"Target duration: {args.duration}s")
        print(f"Overlap: {args.overlap}s")
        print(f"Backup enabled: {not args.no_backup}")

        response = input("\nContinue? (yes/no): ")

        if response.lower() not in ["yes", "y"]:
            print("Cancelled.")
            return

        # Process files
        print("\n" + "=" * 80)
        print("PROCESSING AUDIO FILES")
        print("=" * 80 + "\n")

        files_processed, total_chunks = shortener.process_directory(directory, recursive=not args.no_recursive)

        if files_processed > 0:
            print("\n✅ Processing complete!")
            print(f"Original files backed up to: {directory}/backup_original")
    else:
        print("\n✅ All files are already optimal duration. No processing needed.")


if __name__ == "__main__":
    main()
