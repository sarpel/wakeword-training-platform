"""
Batch Feature Extraction for NPY Generation
Extracts features from entire datasets and saves as .npy files
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import structlog
import torch
from tqdm import tqdm

from src.config.defaults import AugmentationConfig, DataConfig
from src.data.audio_utils import AudioProcessor
from src.data.augmentation import AudioAugmentation
from src.data.feature_extraction import FeatureExtractor

logger = structlog.get_logger(__name__)


class BatchFeatureExtractor:
    """
    Batch extract and save features for entire dataset
    """

    def __init__(self, config: DataConfig, device: str = "cuda"):
        """
        Initialize batch feature extractor

        Args:
            config: DataConfig with feature extraction parameters
            device: Device for computation (cuda or cpu)
        """
        self.config = config
        self.device = device if torch.cuda.is_available() else "cpu"

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            sample_rate=config.sample_rate,
            feature_type=config.feature_type,  # type: ignore
            n_mels=config.n_mels,
            n_mfcc=config.n_mfcc,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            device=self.device,
        )

        # Initialize audio processor
        self.audio_processor = AudioProcessor(target_sr=config.sample_rate, target_duration=config.audio_duration)

        logger.info(f"BatchFeatureExtractor initialized on {self.device}")

    def extract_dataset(
        self,
        audio_files: List[Path],
        output_dir: Path,
        batch_size: int = 32,
        preserve_structure: bool = True,
        progress_callback: Optional[Callable] = None,
        # NEW: Multi-augmentation parameters
        augmentation_multiplier: int = 1,
        augmentation_config: Optional[AugmentationConfig] = None,
        background_noise_dir: Optional[Path] = None,
        rir_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Extract features for all audio files with optional multi-augmentation.

        Args:
            audio_files: List of audio file paths
            output_dir: Directory to save .npy files
            batch_size: Batch size for GPU processing
            preserve_structure: Preserve directory structure in output
            progress_callback: Progress callback(current, total, message)
            augmentation_multiplier: Number of versions per file (1 = original only)
            augmentation_config: Augmentation configuration (required if multiplier > 1)
            background_noise_dir: Directory containing background noise files
            rir_dir: Directory containing RIR files

        Returns:
            Dictionary with extraction results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate total files including augmented versions
        total_versions = len(audio_files) * augmentation_multiplier

        results: Dict[str, Any] = {
            "total_files": len(audio_files),
            "total_versions": total_versions,
            "success_count": 0,
            "failed_count": 0,
            "failed_files": [],
            "augmentation_multiplier": augmentation_multiplier,
        }

        # Initialize augmentation if multiplier > 1
        augmenter: Optional[AudioAugmentation] = None
        if augmentation_multiplier > 1:
            if augmentation_config is None:
                augmentation_config = AugmentationConfig()

            # Collect background noise and RIR files
            background_files = None
            rir_files = None

            if background_noise_dir and Path(background_noise_dir).exists():
                background_files = list(Path(background_noise_dir).rglob("*.wav")) + list(
                    Path(background_noise_dir).rglob("*.mp3")
                )
                logger.info(f"Found {len(background_files)} background noise files for augmentation")

            if rir_dir and Path(rir_dir).exists():
                rir_files = (
                    list(Path(rir_dir).rglob("*.wav"))
                    + list(Path(rir_dir).rglob("*.flac"))
                    + list(Path(rir_dir).rglob("*.mp3"))
                )
                logger.info(f"Found {len(rir_files)} RIR files for augmentation")

            augmenter = AudioAugmentation(
                sample_rate=self.config.sample_rate,
                device=self.device,
                time_stretch_range=(augmentation_config.time_stretch_min, augmentation_config.time_stretch_max),
                pitch_shift_range=(augmentation_config.pitch_shift_min, augmentation_config.pitch_shift_max),
                time_shift_prob=augmentation_config.time_shift_prob,
                time_shift_range_ms=(augmentation_config.time_shift_min_ms, augmentation_config.time_shift_max_ms),
                background_noise_prob=augmentation_config.background_noise_prob,
                noise_snr_range=(augmentation_config.noise_snr_min, augmentation_config.noise_snr_max),
                rir_prob=augmentation_config.rir_prob,
                rir_dry_wet_min=augmentation_config.rir_dry_wet_min,
                rir_dry_wet_max=augmentation_config.rir_dry_wet_max,
                background_noise_files=background_files,
                rir_files=rir_files,
            )
            logger.info(f"Augmentation enabled: {augmentation_multiplier} versions per file")

        logger.info(f"Extracting features for {len(audio_files)} files (total {total_versions} versions)...")

        # Process in batches
        processed_count = 0
        for i in tqdm(range(0, len(audio_files), batch_size), desc="Extracting features"):
            batch_files = audio_files[i : i + batch_size]

            # Load audio batch
            audio_batch = []
            valid_indices = []
            raw_audios: List[np.ndarray] = []  # Keep raw audio for augmentation

            for idx, audio_file in enumerate(batch_files):
                try:
                    # Load and process audio
                    audio = self.audio_processor.process_audio(audio_file)
                    audio_batch.append(audio)
                    raw_audios.append(audio)
                    valid_indices.append(idx)
                except Exception as e:
                    logger.error(f"Failed to load {audio_file}: {e}")
                    results["failed_count"] += augmentation_multiplier  # Count all versions as failed
                    results["failed_files"].append({"path": str(audio_file), "error": str(e)})

            if not audio_batch:
                continue

            # Convert to tensor batch
            audio_tensor = torch.from_numpy(np.stack(audio_batch)).float().to(self.device)

            # Extract features for ORIGINAL audio
            with torch.no_grad():
                features_batch = self.feature_extractor(audio_tensor)

            # Move back to CPU for saving
            features_batch = features_batch.cpu().numpy()

            # Save original feature files
            for idx, valid_idx in enumerate(valid_indices):
                audio_file = batch_files[valid_idx]
                features = features_batch[idx]
                output_path = self._get_output_path(audio_file, output_dir, preserve_structure)

                if self._save_features(output_path, features, audio_file, results):
                    processed_count += 1

            # Extract features for AUGMENTED versions (if multiplier > 1)
            if augmentation_multiplier > 1 and augmenter is not None:
                for aug_idx in range(1, augmentation_multiplier):
                    augmented_batch = []

                    for idx, valid_idx in enumerate(valid_indices):
                        audio_file = batch_files[valid_idx]
                        raw_audio = raw_audios[idx]

                        # Create deterministic seed from file path + aug index
                        seed = hash(str(audio_file) + str(aug_idx)) % (2**31)

                        # Apply augmentation with deterministic seed
                        audio_tensor_single = torch.from_numpy(raw_audio).float().to(self.device)
                        augmented = augmenter.augment_for_extraction(audio_tensor_single, seed=seed)
                        augmented_batch.append(augmented.cpu().numpy())

                    # Stack and extract features for augmented batch
                    aug_tensor = torch.from_numpy(np.stack(augmented_batch)).float().to(self.device)

                    with torch.no_grad():
                        aug_features_batch = self.feature_extractor(aug_tensor)

                    aug_features_batch = aug_features_batch.cpu().numpy()

                    # Save augmented feature files
                    for idx, valid_idx in enumerate(valid_indices):
                        audio_file = batch_files[valid_idx]
                        features = aug_features_batch[idx]
                        output_path = self._get_output_path(
                            audio_file, output_dir, preserve_structure, aug_suffix=f"_aug{aug_idx}"
                        )

                        if self._save_features(output_path, features, audio_file, results):
                            processed_count += 1

            # Update progress
            if progress_callback:
                current = min(i + batch_size, len(audio_files))
                msg = f"{results['success_count']} extracted, {results['failed_count']} failed"
                progress_callback(current, len(audio_files), msg)

        logger.info(
            f"Extraction complete: {results['success_count']} success, "
            f"{results['failed_count']} failed (multiplier={augmentation_multiplier})"
        )

        return results

    def _get_output_path(
        self, audio_file: Path, output_dir: Path, preserve_structure: bool, aug_suffix: str = ""
    ) -> Path:
        """Get output path for a feature file, optionally with augmentation suffix."""
        if preserve_structure:
            try:
                # Find category in path
                parts = audio_file.parts
                category_idx = None
                for i, part in enumerate(parts):
                    if part in ["positive", "negative", "hard_negative"]:
                        category_idx = i
                        break

                if category_idx is not None:
                    relative_parts = parts[category_idx:]
                    output_path = output_dir.joinpath(*relative_parts)
                    # Add augmentation suffix before extension
                    output_path = output_path.with_suffix("")  # Remove .wav/.mp3
                    output_path = Path(str(output_path) + aug_suffix + ".npy")
                else:
                    output_path = output_dir / (audio_file.stem + aug_suffix + ".npy")
            except Exception as e:
                logger.warning(f"Path extraction failed for {audio_file}: {e}")
                output_path = output_dir / (audio_file.stem + aug_suffix + ".npy")
        else:
            output_path = output_dir / (audio_file.stem + aug_suffix + ".npy")

        return output_path

    def _save_features(
        self, output_path: Path, features: np.ndarray, audio_file: Path, results: Dict[str, Any]
    ) -> bool:
        """Save features to file and update results. Returns True on success."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            np.save(str(output_path), features)
            results["success_count"] += 1
            return True
        except Exception as e:
            logger.error(f"Failed to save {output_path}: {e}")
            results["failed_count"] += 1
            results["failed_files"].append({"path": str(audio_file), "error": f"Save failed: {str(e)}"})
            return False

    def extract_from_manifest(
        self,
        manifest_files: List[Path],
        output_dir: Path,
        batch_size: int = 32,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
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
            with open(manifest_file, "r") as f:
                manifest = json.load(f)

            for file_info in manifest["files"]:
                all_audio_files.append(Path(file_info["path"]))

        logger.info(f"Found {len(all_audio_files)} files across {len(manifest_files)} manifests")

        # Extract features
        return self.extract_dataset(
            audio_files=all_audio_files,
            output_dir=output_dir,
            batch_size=batch_size,
            preserve_structure=True,
            progress_callback=progress_callback,
        )


if __name__ == "__main__":
    print("Batch Feature Extractor Test")
    print("=" * 60)

    # Test with default config
    config = DataConfig()

    extractor = BatchFeatureExtractor(config=config, device="cuda" if torch.cuda.is_available() else "cpu")

    print(f"Extractor initialized on: {extractor.device}")
    print("Batch Feature Extractor module loaded successfully")
