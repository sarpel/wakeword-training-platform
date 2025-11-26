"""
PyTorch Dataset for Wakeword Training
Handles audio loading, preprocessing, and augmentation
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import structlog

from src.data.balanced_sampler import create_balanced_sampler_from_dataset
from src.data.cmvn import compute_cmvn_from_dataset, CMVN
from src.data.audio_utils import AudioProcessor
from src.data.feature_extraction import FeatureExtractor
from src.data.augmentation import AudioAugmentation

logger = structlog.get_logger(__name__)


class WakewordDataset(Dataset):
    """
    PyTorch Dataset for wakeword detection

    Loads audio files and applies preprocessing
    """

    def __init__(
        self,
        manifest_path: Path,
        sample_rate: int = 16000,
        audio_duration: float = 1.5,
        augment: bool = False,
        cache_audio: bool = False,
        augmentation_config: Optional[Dict] = None,
        background_noise_dir: Optional[Path] = None,
        rir_dir: Optional[Path] = None,
        device: str = 'cpu',  # Dataset operations always on CPU
        feature_type: str = 'mel',
        n_mels: int = 64,
        n_mfcc: int = 0,
        n_fft: int = 400,
        hop_length: int = 160,
        # NEW: NPY feature parameters
        use_precomputed_features_for_training: bool = True,
        npy_cache_features: bool = True,
        fallback_to_audio: bool = False,
        # CMVN parameters
        cmvn_path: Optional[Path] = None,
        apply_cmvn: bool = False
    ):
        """
        Initialize wakeword dataset

        Args:
            manifest_path: Path to split manifest JSON (train.json, val.json, test.json)
            sample_rate: Target sample rate
            audio_duration: Target audio duration in seconds
            augment: Apply data augmentation
            cache_audio: Cache loaded audio in memory (use for small datasets)
            augmentation_config: Configuration for augmentation (time_stretch_range, pitch_shift_range, etc.)
            background_noise_dir: Directory containing background noise files
            rir_dir: Directory containing RIR files
            device: Device for feature extraction (always 'cpu' for dataset pipeline)
            feature_type: Feature type ('mel' or 'mfcc')
            n_mels: Number of mel filterbanks
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for STFT
            use_precomputed_features_for_training: Enable loading from .npy files
            npy_cache_features: Cache loaded .npy features in RAM
            fallback_to_audio: If NPY missing, load raw audio
            cmvn_path: Path to CMVN stats.json file
            apply_cmvn: Whether to apply CMVN normalization
        """
        self.manifest_path = Path(manifest_path)
        self.sample_rate = sample_rate
        self.audio_duration = audio_duration
        self.augment = augment
        self.cache_audio = cache_audio
        self.device = 'cpu'  # Always CPU for dataset operations

        # NEW: NPY feature parameters
        self.use_precomputed_features_for_training = use_precomputed_features_for_training
        self.npy_cache_features = npy_cache_features
        self.fallback_to_audio = fallback_to_audio

        # CMVN initialization
        self.apply_cmvn = apply_cmvn
        self.cmvn = None
        if apply_cmvn and cmvn_path and Path(cmvn_path).exists():
            self.cmvn = CMVN(stats_path=cmvn_path)
            logger.info(f"CMVN loaded from {cmvn_path}")
        elif apply_cmvn:
            logger.warning(f"CMVN requested but stats file not found: {cmvn_path}")

        # Load manifest
        with open(self.manifest_path, 'r') as f:
            manifest = json.load(f)

        self.files = manifest['files']
        self.categories = manifest['categories']

        # Create label mapping
        self.label_map = self._create_label_map()

        # Audio processor
        self.audio_processor = AudioProcessor(
            target_sr=sample_rate,
            target_duration=audio_duration
        )

        # Cache for loaded audio
        self.audio_cache = {} if cache_audio else None

        # NEW: Feature cache (separate from audio cache)
        self.feature_cache = {} if npy_cache_features else None

        # Normalize feature type (handle legacy 'mel_spectrogram')
        if feature_type == 'mel_spectrogram':
            feature_type = 'mel'

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            sample_rate=sample_rate,
            feature_type=feature_type,
            n_mels=n_mels,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            device=device
        )

        # Initialize augmentation if enabled
        self.augmentation = None
        if augment:
            # Collect background noise and RIR files
            background_files = None
            rir_files = None

            if background_noise_dir and Path(background_noise_dir).exists():
                background_files = list(Path(background_noise_dir).rglob("*.wav")) + \
                                 list(Path(background_noise_dir).rglob("*.mp3"))
                logger.info(f"Found {len(background_files)} background noise files")

            if rir_dir and Path(rir_dir).exists():
                rir_files = list(Path(rir_dir).rglob("*.wav")) + \
                            list(Path(rir_dir).rglob("*.flac")) + \
                            list(Path(rir_dir).rglob("*.mp3"))
                logger.info(f"Found {len(rir_files)} RIR files")

            # Get augmentation parameters from config or use defaults
            aug_config = augmentation_config or {}

            self.augmentation = AudioAugmentation(
                sample_rate=sample_rate,
                device='cpu',  # Audio augmentation always on CPU
                time_stretch_range=aug_config.get('time_stretch_range', (0.9, 1.1)),
                pitch_shift_range=aug_config.get('pitch_shift_range', (-2, 2)),
                background_noise_prob=aug_config.get('background_noise_prob', 0.5),
                noise_snr_range=aug_config.get('noise_snr_range', (5.0, 20.0)),
                rir_prob=aug_config.get('rir_prob', 0.25),
                rir_dry_wet_min=aug_config.get('rir_dry_wet_min', 0.3),
                rir_dry_wet_max=aug_config.get('rir_dry_wet_max', 0.7),
                background_noise_files=background_files,
                rir_files=rir_files
            )

            logger.info("Augmentation pipeline initialized")

        logger.info(
            f"Dataset initialized: {len(self.files)} files, "
            f"categories: {self.categories}, augment: {augment}"
        )

    def _create_label_map(self) -> Dict[str, int]:
        """
        Create category to label mapping

        Returns:
            Dictionary mapping category names to integer labels
        """
        # Binary classification: positive (1) vs everything else (0)
        # Note: background and rirs are NOT training samples - they're augmentation sources
        label_map = {
            'positive': 1,
            'negative': 0,
            'hard_negative': 0
        }

        return label_map

    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.files)

    def _load_from_npy(self, file_info: Dict, idx: int) -> Optional[torch.Tensor]:
        """
        Load precomputed features from .npy file

        Args:
            file_info: File metadata from manifest
            idx: Sample index (for caching)

        Returns:
            Feature tensor or None if not found
        """
        # Check cache first
        if self.feature_cache is not None and idx in self.feature_cache:
            return self.feature_cache[idx]

        # Get NPY path from manifest
        npy_path = file_info.get('npy_path')

        if not npy_path or not Path(npy_path).exists():
            return None

        try:
            # Load NPY file (memory-mapped for efficiency)
            features = np.load(npy_path, mmap_mode='r')

            # Convert to tensor
            features_tensor = torch.from_numpy(np.array(features)).float()

            # Validate shape
            expected_shape = self.feature_extractor.get_output_shape(
                int(self.audio_duration * self.sample_rate)
            )

            if features_tensor.shape != expected_shape:
                logger.warning(
                    f"Shape mismatch for {npy_path}: "
                    f"expected {expected_shape}, got {features_tensor.shape}"
                )
                return None

            # Cache if enabled
            if self.feature_cache is not None:
                self.feature_cache[idx] = features_tensor

            return features_tensor

        except Exception as e:
            logger.error(f"Error loading NPY {npy_path}: {e}")
            return None

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Get item by index

        Args:
            idx: Sample index

        Returns:
            Tuple of (features_tensor, label, metadata)
        """
        file_info = self.files[idx]
        file_path = Path(file_info['path'])
        category = file_info['category']
        label = self.label_map[category]

        # NEW: Try loading from NPY first if enabled
        if self.use_precomputed_features_for_training:
            features = self._load_from_npy(file_info, idx)

            if features is not None:
                # Apply CMVN if enabled
                if self.cmvn is not None:
                    features = self.cmvn.normalize(features)

                # Successfully loaded from NPY
                metadata = {
                    'path': str(file_path),
                    'category': category,
                    'label': label,
                    'source': 'npy',  # NEW: Track data source
                    'sample_rate': self.sample_rate,
                    'duration': self.audio_duration
                }
                return features, label, metadata

            elif not self.fallback_to_audio:
                raise FileNotFoundError(
                    f"NPY file not found for {file_path} and fallback disabled"
                )

            # If NPY not found, fall through to audio loading

        # EXISTING: Load from raw audio
        if self.audio_cache is not None and idx in self.audio_cache:
            audio = self.audio_cache[idx]
        else:
            # Process audio (load, resample, normalize length)
            audio = self.audio_processor.process_audio(file_path)

            if self.audio_cache is not None:
                self.audio_cache[idx] = audio

        # Apply augmentation if enabled
        if self.augment:
            audio = self._apply_augmentation(audio)

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()

        # Extract features (mel-spectrogram or MFCC) on CPU
        features = self.feature_extractor(audio_tensor)

        # Apply CMVN if enabled
        if self.cmvn is not None:
            features = self.cmvn.normalize(features)

        # Metadata
        metadata = {
            'path': str(file_path),
            'category': category,
            'label': label,
            'source': 'audio',  # NEW: Track data source
            'sample_rate': self.sample_rate,
            'duration': self.audio_duration
        }

        return features, label, metadata

    def _apply_augmentation(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation using AudioAugmentation pipeline

        Args:
            audio: Input audio array (1D numpy array)

        Returns:
            Augmented audio array
        """
        if self.augmentation is None:
            return audio

        # Convert to torch tensor
        # Add channel dimension if needed
        if audio.ndim == 1:
            audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()  # (1, samples)
        else:
            audio_tensor = torch.from_numpy(audio).float()

        # Audio augmentation runs on CPU (device is always 'cpu')
        # Apply augmentation
        augmented_tensor = self.augmentation(audio_tensor)

        # Convert back to numpy (already on CPU)
        augmented = augmented_tensor.squeeze(0).numpy()

        return augmented

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced datasets with guards for edge cases

        Returns:
            Tensor of class weights
        """
        # Count samples per class
        labels = [self.label_map[f['category']] for f in self.files]

        if len(labels) == 0:
            logger.warning("No samples found, returning equal weights")
            return torch.ones(2)

        label_counts = np.bincount(labels)

        # Guard: Check if any class has zero samples
        if np.any(label_counts == 0):
            logger.warning("Zero samples in one or more classes, returning equal weights")
            return torch.ones(len(label_counts))

        # Guard: Check if only one class exists
        if len(label_counts) == 1:
            logger.warning("Only one class present, returning equal weights")
            return torch.ones(len(label_counts))

        # Calculate weights (inverse frequency)
        total_samples = len(labels)
        class_weights = total_samples / (len(label_counts) * label_counts)

        return torch.from_numpy(class_weights).float()


def load_dataset_splits(
    data_root: Path,
    sample_rate: int = 16000,
    audio_duration: float = 1.5,
    augment_train: bool = True,
    cache_audio: bool = False,
    augmentation_config: Optional[Dict] = None,
    device: str = 'cpu',  # Dataset operations always on CPU
    feature_type: str = 'mel',
    n_mels: int = 64,
    n_mfcc: int = 0,
    n_fft: int = 400,
    hop_length: int = 160,
    # NEW: NPY feature parameters
    use_precomputed_features_for_training: bool = True,
    npy_cache_features: bool = True,
    fallback_to_audio: bool = False,
    # CMVN parameters
    cmvn_path: Optional[Path] = None,
    apply_cmvn: bool = False
) -> Tuple[WakewordDataset, WakewordDataset, WakewordDataset]:
    """
    Load train, validation, and test datasets

    Args:
        data_root: Root directory containing data
        sample_rate: Target sample rate
        audio_duration: Target audio duration
        augment_train: Apply augmentation to training set
        cache_audio: Cache loaded audio in memory
        augmentation_config: Configuration for augmentation
        device: Device for feature extraction (always 'cpu' for dataset pipeline)
        feature_type: Feature type ('mel' or 'mfcc')
        n_mels: Number of mel filterbanks
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length for STFT
        use_precomputed_features_for_training: Enable loading from .npy files
        npy_cache_features: Cache loaded .npy features in RAM
        fallback_to_audio: If NPY missing, load raw audio
        cmvn_path: Path to CMVN stats.json file
        apply_cmvn: Whether to apply CMVN normalization

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    splits_dir = data_root / "splits"

    # Force CPU for dataset operations
    device = 'cpu'

    # Determine background noise and RIR directories
    background_noise_dir = data_root / "raw" / "background"
    rir_dir = data_root / "raw" / "rirs"

    train_dataset = WakewordDataset(
        manifest_path=splits_dir / "train.json",
        sample_rate=sample_rate,
        audio_duration=audio_duration,
        augment=augment_train,
        cache_audio=cache_audio,
        augmentation_config=augmentation_config,
        background_noise_dir=background_noise_dir,
        rir_dir=rir_dir,
        device=device,
        feature_type=feature_type,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        use_precomputed_features_for_training=use_precomputed_features_for_training,
        npy_cache_features=npy_cache_features,
        fallback_to_audio=fallback_to_audio,
        cmvn_path=cmvn_path,
        apply_cmvn=apply_cmvn
    )

    val_dataset = WakewordDataset(
        manifest_path=splits_dir / "val.json",
        sample_rate=sample_rate,
        audio_duration=audio_duration,
        augment=False,  # No augmentation for validation
        cache_audio=cache_audio,
        device=device,
        feature_type=feature_type,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        use_precomputed_features_for_training=use_precomputed_features_for_training,
        npy_cache_features=npy_cache_features,
        fallback_to_audio=fallback_to_audio,
        cmvn_path=cmvn_path,
        apply_cmvn=apply_cmvn
    )

    test_dataset = WakewordDataset(
        manifest_path=splits_dir / "test.json",
        sample_rate=sample_rate,
        audio_duration=audio_duration,
        augment=False,  # No augmentation for test
        cache_audio=cache_audio,
        device=device,
        feature_type=feature_type,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        use_precomputed_features_for_training=use_precomputed_features_for_training,
        npy_cache_features=npy_cache_features,
        fallback_to_audio=fallback_to_audio,
        cmvn_path=cmvn_path,
        apply_cmvn=apply_cmvn
    )

    logger.info(
        f"Datasets loaded: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Test dataset
    print("Wakeword Dataset Test")
    print("=" * 60)

    # This would test if splits exist
    data_root = Path("data")

    if (data_root / "splits").exists() and (data_root / "splits" / "train.json").exists():
        try:
            train_ds, val_ds, test_ds = load_dataset_splits(data_root)
            print(f"Train dataset: {len(train_ds)} samples")
            print(f"Val dataset: {len(val_ds)} samples")
            print(f"Test dataset: {len(test_ds)} samples")

            # Test loading a sample
            if len(train_ds) > 0:
                audio, label, metadata = train_ds[0]
                print(f"\nSample:")
                print(f"  Audio shape: {audio.shape}")
                print(f"  Label: {label}")
                print(f"  Metadata: {metadata}")

            print("\nDataset test complete")
        except Exception as e:
            print(f"Error loading datasets: {e}")
            print("Make sure to run dataset splitting first")
    else:
        print(f"Splits directory not found: {splits_dir}")
        print("Run Panel 1 to scan and split datasets first")

    print("\nDataset module loaded successfully")