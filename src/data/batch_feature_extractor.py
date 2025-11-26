"""
Batch Feature Extraction for NPY Generation
Extracts features from entire datasets and saves as .npy files
"""
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Callable
import structlog
from tqdm import tqdm

from src.data.audio_utils import AudioProcessor
from src.data.feature_extraction import FeatureExtractor
from src.config.defaults import DataConfig

logger = structlog.get_logger(__name__)


class BatchFeatureExtractor:
    """
    Batch extract and save features for entire dataset
    """

    def __init__(
        self,
        config: DataConfig,
        device: str = 'cuda'
    ):
        """
        Initialize batch feature extractor

        Args:
            config: DataConfig with feature extraction parameters
            device: Device for computation (cuda or cpu)
        """
        self.config = config
        self.device = device if torch.cuda.is_available() else 'cpu'

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            sample_rate=config.sample_rate,
            feature_type=config.feature_type,
            n_mels=config.n_mels,
            n_mfcc=config.n_mfcc,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            device=self.device
        )

        # Initialize audio processor
        self.audio_processor = AudioProcessor(
            target_sr=config.sample_rate,
            target_duration=config.audio_duration
        )

        logger.info(f"BatchFeatureExtractor initialized on {self.device}")

    def extract_dataset(
        self,
        audio_files: List[Path],
        output_dir: Path,
        batch_size: int = 32,
        preserve_structure: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Extract features for all audio files

        Args:
            audio_files: List of audio file paths
            output_dir: Directory to save .npy files
            batch_size: Batch size for GPU processing
            preserve_structure: Preserve directory structure in output
            progress_callback: Progress callback(current, total, message)

        Returns:
            Dictionary with extraction results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            'total_files': len(audio_files),
            'success_count': 0,
            'failed_count': 0,
            'failed_files': []
        }

        logger.info(f"Extracting features for {len(audio_files)} files...")

        # Process in batches
        for i in tqdm(range(0, len(audio_files), batch_size), desc="Extracting features"):
            batch_files = audio_files[i:i + batch_size]

            # Load audio batch
            audio_batch = []
            valid_indices = []

            for idx, audio_file in enumerate(batch_files):
                try:
                    # Load and process audio
                    audio = self.audio_processor.process_audio(audio_file)
                    audio_batch.append(audio)
                    valid_indices.append(idx)
                except Exception as e:
                    logger.error(f"Failed to load {audio_file}: {e}")
                    results['failed_count'] += 1
                    results['failed_files'].append({
                        'path': str(audio_file),
                        'error': str(e)
                    })

            if not audio_batch:
                continue

            # Convert to tensor batch
            audio_tensor = torch.from_numpy(np.stack(audio_batch)).float().to(self.device)

            # Extract features (on GPU if available)
            with torch.no_grad():
                features_batch = self.feature_extractor(audio_tensor)

            # Move back to CPU for saving
            features_batch = features_batch.cpu().numpy()

            # Save individual feature files
            for idx, valid_idx in enumerate(valid_indices):
                audio_file = batch_files[valid_idx]
                features = features_batch[idx]

                # Determine output path
                if preserve_structure:
                    # Preserve full directory structure starting from category
                    try:
                        # Find category in path (positive, negative, hard_negative)
                        parts = audio_file.parts
                        category_idx = None
                        for i, part in enumerate(parts):
                            if part in ['positive', 'negative', 'hard_negative']:
                                category_idx = i
                                break

                        if category_idx is not None:
                            # Preserve everything from category onwards
                            # Example: data/raw/positive/hey_cut/en/file.wav
                            #       -> data/raw/npy/positive/hey_cut/en/file.npy
                            relative_parts = parts[category_idx:]  # positive/hey_cut/en/file.wav
                            output_path = output_dir.joinpath(*relative_parts).with_suffix('.npy')
                        else:
                            # Fallback: just use filename
                            output_path = output_dir / (audio_file.stem + '.npy')
                    except Exception as e:
                        logger.warning(f"Path extraction failed for {audio_file}: {e}")
                        output_path = output_dir / (audio_file.stem + '.npy')
                else:
                    output_path = output_dir / (audio_file.stem + '.npy')

                # Create parent directory
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Save features
                try:
                    np.save(str(output_path), features)
                    results['success_count'] += 1
                except Exception as e:
                    logger.error(f"Failed to save {output_path}: {e}")
                    results['failed_count'] += 1
                    results['failed_files'].append({
                        'path': str(audio_file),
                        'error': f"Save failed: {str(e)}"
                    })

            # Update progress
            if progress_callback:
                current = min(i + batch_size, len(audio_files))
                msg = f"{results['success_count']} extracted, {results['failed_count']} failed"
                progress_callback(current, len(audio_files), msg)

        logger.info(
            f"Extraction complete: {results['success_count']} success, "
            f"{results['failed_count']} failed"
        )

        return results

    def extract_from_manifest(
        self,
        manifest_files: List[Path],
        output_dir: Path,
        batch_size: int = 32,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Extract features from dataset manifest files

        Args:
            manifest_files: List of manifest JSON files (train.json, val.json, test.json)
            output_dir: Directory to save .npy files
            batch_size: Batch size for GPU processing
            progress_callback: Progress callback(current, total, message)

        Returns:
            Dictionary with extraction results
        """
        import json

        # Collect all audio files from manifests
        all_audio_files = []

        for manifest_file in manifest_files:
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)

            for file_info in manifest['files']:
                all_audio_files.append(Path(file_info['path']))

        logger.info(f"Found {len(all_audio_files)} files across {len(manifest_files)} manifests")

        # Extract features
        return self.extract_dataset(
            audio_files=all_audio_files,
            output_dir=output_dir,
            batch_size=batch_size,
            preserve_structure=True,
            progress_callback=progress_callback
        )


if __name__ == "__main__":
    print("Batch Feature Extractor Test")
    print("=" * 60)

    # Test with default config
    config = DataConfig()

    extractor = BatchFeatureExtractor(
        config=config,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"Extractor initialized on: {extractor.device}")
    print("Batch Feature Extractor module loaded successfully")
