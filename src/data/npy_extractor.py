"""
.npy File Extractor and Processor
Handles loading, validation, and conversion of .npy feature files
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from tqdm import tqdm
import structlog
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = structlog.get_logger(__name__)


class NpyExtractor:
    """Extract and process .npy feature files"""

    SUPPORTED_SHAPES = {
        'raw_audio': 1,        # (N, samples)
        'spectrogram': 2,      # (N, freq_bins, time_steps)
        'mfcc': 2,             # (N, n_mfcc, time_steps)
        'features_2d': 2,      # Generic 2D features
        'features_3d': 3,      # Generic 3D features
    }

    def __init__(self, max_workers: int = None):
        """
        Initialize NPY extractor

        Args:
            max_workers: Maximum number of parallel workers (default: CPU count)
        """
        self.extracted_files = []
        self.errors = []

        # Set max workers for parallel processing
        if max_workers is None:
            max_workers = max(multiprocessing.cpu_count() - 2, 1)
        self.max_workers = max_workers

    def scan_npy_files(self, directory: Path, recursive: bool = True) -> List[Path]:
        """
        Scan directory for .npy files

        Args:
            directory: Directory to scan
            recursive: Search recursively

        Returns:
            List of .npy file paths
        """
        directory = Path(directory)

        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return []

        if recursive:
            npy_files = list(directory.rglob("*.npy"))
        else:
            npy_files = list(directory.glob("*.npy"))

        logger.info(f"Found {len(npy_files)} .npy files in {directory}")
        return sorted(npy_files)

    def validate_npy_file(self, file_path: Path) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Validate .npy file and extract metadata

        Args:
            file_path: Path to .npy file

        Returns:
            Tuple of (is_valid, metadata, error_message)
        """
        try:
            # Load array (memory-mapped for efficiency)
            data = np.load(str(file_path), mmap_mode='r')

            # Get metadata
            metadata = {
                'path': str(file_path),
                'filename': file_path.name,
                'shape': data.shape,
                'ndim': data.ndim,
                'dtype': str(data.dtype),
                'size': data.size,
                'nbytes': data.nbytes,
                'nbytes_mb': data.nbytes / (1024 ** 2)
            }

            # Determine feature type
            feature_type = self._infer_feature_type(data.shape, data.ndim)
            metadata['inferred_type'] = feature_type

            # Basic validation
            if data.size == 0:
                return False, None, "Empty array"

            if not np.isfinite(data).all():
                return False, None, "Contains NaN or Inf values"

            return True, metadata, None

        except Exception as e:
            error_msg = f"Error loading {file_path.name}: {str(e)}"
            logger.debug(error_msg)
            return False, None, error_msg

    def _infer_feature_type(self, shape: Tuple, ndim: int) -> str:
        """
        Infer feature type from shape

        Args:
            shape: Array shape
            ndim: Number of dimensions

        Returns:
            Inferred feature type string
        """
        if ndim == 1:
            return "raw_audio_1d"
        elif ndim == 2:
            # Could be raw audio (N, samples) or features (N, features)
            if shape[1] > 1000:
                return "raw_audio_batch"
            elif shape[0] in [13, 20, 40, 80, 128]:  # Common MFCC/mel dimensions
                return "mfcc_or_spectrogram"
            else:
                return "features_2d"
        elif ndim == 3:
            # (N, freq_bins, time_steps) or (N, n_mfcc, time_steps)
            return "spectrogram_batch"
        else:
            return f"unknown_{ndim}d"

    def extract_and_convert(
        self,
        npy_files: List[Path],
        output_format: str = 'audio',
        target_sr: int = 16000,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Extract and convert .npy files with parallel processing

        Args:
            npy_files: List of .npy file paths
            output_format: Output format ('audio', 'keep', 'normalize')
            target_sr: Target sample rate if converting to audio
            progress_callback: Optional progress callback(current, total, message)

        Returns:
            Dictionary with extraction results
        """
        results = {
            'total_files': len(npy_files),
            'valid_files': 0,
            'invalid_files': 0,
            'files': [],
            'errors': []
        }

        logger.info(f"Extracting {len(npy_files)} .npy files with {self.max_workers} workers...")

        completed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all validation tasks
            future_to_file = {
                executor.submit(self.validate_npy_file, file_path): file_path
                for file_path in npy_files
            }

            # Process completed validations
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                completed += 1

                try:
                    is_valid, metadata, error = future.result()

                    if is_valid:
                        results['valid_files'] += 1
                        results['files'].append(metadata)
                    else:
                        results['invalid_files'] += 1
                        results['errors'].append({
                            'path': str(file_path),
                            'error': error
                        })

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    results['invalid_files'] += 1
                    results['errors'].append({
                        'path': str(file_path),
                        'error': str(e)
                    })

                # Update progress
                if progress_callback:
                    msg = f"{completed}/{len(npy_files)} files"
                    progress_callback(completed, len(npy_files), msg)

        self.extracted_files = results['files']
        self.errors = results['errors']

        logger.info(
            f"Extraction complete: {results['valid_files']} valid, "
            f"{results['invalid_files']} invalid"
        )

        return results

    def get_statistics(self) -> Dict:
        """
        Get statistics about extracted .npy files

        Returns:
            Dictionary with statistics
        """
        if not self.extracted_files:
            return {'status': 'No files extracted yet'}

        # Count feature types
        feature_types = {}
        dtypes = {}
        total_size_mb = 0.0

        for file_info in self.extracted_files:
            feature_type = file_info['inferred_type']
            dtype = file_info['dtype']

            feature_types[feature_type] = feature_types.get(feature_type, 0) + 1
            dtypes[dtype] = dtypes.get(dtype, 0) + 1
            total_size_mb += file_info['nbytes_mb']

        return {
            'total_files': len(self.extracted_files),
            'total_size_mb': round(total_size_mb, 2),
            'feature_types': feature_types,
            'dtypes': dtypes,
            'avg_size_mb': round(total_size_mb / len(self.extracted_files), 2) if self.extracted_files else 0
        }

    def load_npy_batch(
        self,
        file_path: Path,
        max_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Load .npy file into memory

        Args:
            file_path: Path to .npy file
            max_samples: Maximum number of samples to load

        Returns:
            Tuple of (data_array, metadata)
        """
        try:
            # Load array
            data = np.load(str(file_path))

            # Limit samples if requested
            if max_samples and data.ndim >= 2 and data.shape[0] > max_samples:
                data = data[:max_samples]

            metadata = {
                'shape': data.shape,
                'dtype': str(data.dtype),
                'loaded_samples': data.shape[0] if data.ndim >= 1 else 1
            }

            return data, metadata

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise

    def convert_to_audio(
        self,
        features: np.ndarray,
        feature_type: str,
        target_sr: int = 16000
    ) -> np.ndarray:
        """
        Convert features back to audio (if possible)

        Args:
            features: Feature array
            feature_type: Type of features
            target_sr: Target sample rate

        Returns:
            Audio array

        Note: Only works for raw audio, not for extracted features like MFCC
        """
        if 'raw_audio' in feature_type:
            # Already audio
            return features

        # For spectrograms/MFCC, conversion requires inverse transforms
        # This is complex and lossy, so we skip it for now
        logger.warning(
            f"Cannot convert {feature_type} to audio without information loss. "
            "Use raw audio files for training instead."
        )
        raise NotImplementedError(f"Conversion from {feature_type} to audio not implemented")

    def generate_report(self) -> str:
        """
        Generate extraction report

        Returns:
            Formatted report string
        """
        stats = self.get_statistics()

        if stats.get('status'):
            return stats['status']

        report = []
        report.append("=" * 60)
        report.append(".NPY FILE EXTRACTION REPORT")
        report.append("=" * 60)
        report.append("")

        report.append(f"Total Files Extracted: {stats['total_files']}")
        report.append(f"Total Size: {stats['total_size_mb']:.2f} MB")
        report.append(f"Average Size: {stats['avg_size_mb']:.2f} MB")
        report.append("")

        report.append("Feature Types:")
        report.append("-" * 60)
        for feature_type, count in stats['feature_types'].items():
            report.append(f"  {feature_type}: {count} files")
        report.append("")

        report.append("Data Types:")
        report.append("-" * 60)
        for dtype, count in stats['dtypes'].items():
            report.append(f"  {dtype}: {count} files")
        report.append("")

        if self.errors:
            report.append(f"Errors: {len(self.errors)}")
            report.append("-" * 60)
            for error in self.errors[:5]:  # Show first 5 errors
                report.append(f"  {error['path']}: {error['error']}")
            if len(self.errors) > 5:
                report.append(f"  ... and {len(self.errors) - 5} more errors")
            report.append("")

        report.append("=" * 60)

        return "\n".join(report)


if __name__ == "__main__":
    # Test NPY extractor
    print("NPY Extractor Test")
    print("=" * 60)

    extractor = NpyExtractor()
    print("NPY Extractor initialized successfully")

    # Test with example data
    test_data = np.random.randn(100, 40, 50)  # Example: 100 samples, 40 MFCC, 50 time steps
    print(f"\nTest data shape: {test_data.shape}")
    print(f"Inferred type: {extractor._infer_feature_type(test_data.shape, test_data.ndim)}")

    print("\nNPY Extractor test complete")