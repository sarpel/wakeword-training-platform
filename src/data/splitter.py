"""
Dataset Scanner, Validator, and Splitter
Handles recursive scanning, validation, statistics, and train/test/val splitting
"""
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from src.data.audio_utils import AudioValidator, scan_audio_files
from src.data.file_cache import FileCache
from src.config.logger import get_data_logger

logger = logging.getLogger(__name__)


class DatasetScanner:
    """Scans and validates audio datasets"""

    CATEGORY_FOLDERS = {
        'positive': 'positive',
        'negative': 'negative',
        'hard_negative': 'hard_negative',
        # Note: 'background' and 'rirs' are NOT included here
        # They are used only for augmentation, loaded separately by AudioAugmentation
        # 'background': 'background',
        # 'rirs': 'rirs'
    }

    def __init__(self, dataset_root: Path, use_cache: bool = True, max_workers: int = None):
        """
        Initialize dataset scanner

        Args:
            dataset_root: Root directory containing dataset folders
            use_cache: Use file cache to speed up scanning
            max_workers: Maximum number of parallel workers (default: CPU count)
        """
        self.dataset_root = Path(dataset_root)
        self.validator = AudioValidator()
        self.dataset_info = {}
        self.statistics = {}
        self.use_cache = use_cache
        self.cache = FileCache() if use_cache else None

        # Set max workers for parallel processing
        if max_workers is None:
            max_workers = max(multiprocessing.cpu_count() - 2, 1)
        self.max_workers = max_workers

    def scan_datasets(self, progress_callback: Optional[Callable] = None, skip_validation: bool = False) -> Dict:
        """
        Scan all dataset categories

        Args:
            progress_callback: Optional callback(current, total, message) for progress updates
            skip_validation: Skip file validation (only count files, much faster)

        Returns:
            Dictionary with scan results
        """
        logger.info(f"Scanning datasets in: {self.dataset_root} (workers={self.max_workers}, cache={self.use_cache})")

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")

        results = {
            'dataset_root': str(self.dataset_root),
            'categories': {},
            'total_files': 0,
            'total_duration': 0.0,
            'valid_files': 0,
            'corrupted_files': 0,
            'warnings': [],
            'cached_files': 0
        }

        # Calculate total files for progress
        total_categories = len([f for f in self.CATEGORY_FOLDERS.values()
                               if (self.dataset_root / f).exists()])
        current_category = 0

        # Scan each category
        for category_key, folder_name in self.CATEGORY_FOLDERS.items():
            category_path = self.dataset_root / folder_name

            if not category_path.exists():
                logger.warning(f"Category folder not found: {category_path}")
                results['warnings'].append(f"Missing folder: {folder_name}")
                continue

            logger.info(f"Scanning category: {category_key}")

            # Create category progress callback
            if progress_callback:
                def category_progress(current, total, msg=""):
                    overall_progress = (current_category / total_categories) + (current / total / total_categories)
                    progress_callback(overall_progress, f"{category_key}: {msg}")

                category_result = self._scan_category(
                    category_path, category_key, category_progress, skip_validation
                )
            else:
                category_result = self._scan_category(
                    category_path, category_key, None, skip_validation
                )

            results['categories'][category_key] = category_result

            results['total_files'] += category_result['total_files']
            results['total_duration'] += category_result['total_duration']
            results['valid_files'] += category_result['valid_files']
            results['corrupted_files'] += category_result['corrupted_files']
            if 'cached_files' in category_result:
                results['cached_files'] += category_result['cached_files']

            current_category += 1

        # Save cache
        if self.cache:
            self.cache.save()
            logger.info(f"Cache saved with {results['cached_files']} cached files")

        self.dataset_info = results
        return results

    def _validate_file(self, file_path: Path) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Validate a single file (with caching support)

        Args:
            file_path: Path to file

        Returns:
            Tuple of (is_valid, metadata, error)
        """
        # Check cache first
        if self.cache:
            cached_metadata = self.cache.get(file_path)
            if cached_metadata:
                return True, cached_metadata, None

        # Validate file
        is_valid, metadata, error = self.validator.validate_audio_file(file_path)

        # Cache valid results
        if is_valid and metadata and self.cache:
            self.cache.set(file_path, metadata)

        return is_valid, metadata, error

    def _scan_category(
        self,
        category_path: Path,
        category_name: str,
        progress_callback: Optional[Callable] = None,
        skip_validation: bool = False
    ) -> Dict:
        """
        Scan a single category folder recursively with parallel processing

        Args:
            category_path: Path to category folder
            category_name: Name of category
            progress_callback: Optional progress callback(current, total, message)
            skip_validation: Skip file validation (only count files)

        Returns:
            Dictionary with category scan results
        """
        # Find all audio files recursively
        audio_files = scan_audio_files(category_path, recursive=True)

        logger.info(f"Found {len(audio_files)} audio files in {category_name}")

        result = {
            'path': str(category_path),
            'total_files': len(audio_files),
            'valid_files': 0,
            'corrupted_files': 0,
            'total_duration': 0.0,
            'sample_rates': defaultdict(int),
            'formats': defaultdict(int),
            'files': [],
            'corrupted': [],
            'quality_warnings': [],
            'cached_files': 0
        }

        if skip_validation:
            # Fast mode: just count files
            result['valid_files'] = len(audio_files)
            result['files'] = [{'path': str(f), 'filename': f.name} for f in audio_files]
            if progress_callback:
                progress_callback(len(audio_files), len(audio_files), "Counted files")
            return result

        # Validate files in parallel
        completed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all validation tasks
            future_to_file = {
                executor.submit(self._validate_file, file_path): file_path
                for file_path in audio_files
            }

            # Process completed validations
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                completed += 1

                try:
                    is_valid, metadata, error = future.result()

                    if is_valid:
                        result['valid_files'] += 1
                        result['total_duration'] += metadata['duration']
                        result['sample_rates'][metadata['sample_rate']] += 1
                        result['formats'][metadata['format']] += 1

                        # Check if from cache
                        if metadata.get('_cached_at'):
                            result['cached_files'] += 1

                        # Check quality
                        quality = self.validator.check_audio_quality(metadata)
                        if quality['warnings']:
                            result['quality_warnings'].extend(quality['warnings'])

                        # Add to file list
                        result['files'].append({
                            'path': metadata['path'],
                            'filename': metadata['filename'],
                            'duration': metadata['duration'],
                            'sample_rate': metadata['sample_rate'],
                            'channels': metadata['channels'],
                            'quality_score': quality['quality_score']
                        })
                    else:
                        result['corrupted_files'] += 1
                        result['corrupted'].append({
                            'path': str(file_path),
                            'error': error
                        })

                except Exception as e:
                    logger.error(f"Error validating {file_path}: {e}")
                    result['corrupted_files'] += 1
                    result['corrupted'].append({
                        'path': str(file_path),
                        'error': str(e)
                    })

                # Update progress
                if progress_callback:
                    msg = f"{completed}/{len(audio_files)} files"
                    if result['cached_files'] > 0:
                        msg += f" ({result['cached_files']} cached)"
                    progress_callback(completed, len(audio_files), msg)

        # Convert defaultdicts to regular dicts for JSON serialization
        result['sample_rates'] = {str(k): v for k, v in result['sample_rates'].items()}
        result['formats'] = {str(k): v for k, v in result['formats'].items()}

        logger.info(f"Scanned {category_name}: {result['valid_files']} valid, "
                   f"{result['corrupted_files']} corrupted, {result['cached_files']} from cache")

        return result

    def get_statistics(self) -> Dict:
        """
        Get dataset statistics

        Returns:
            Dictionary with detailed statistics
        """
        if not self.dataset_info:
            raise ValueError("No dataset scanned yet. Call scan_datasets() first.")

        stats = {
            'total_files': self.dataset_info['valid_files'],
            'total_duration_hours': self.dataset_info['total_duration'] / 3600,
            'total_duration_minutes': self.dataset_info['total_duration'] / 60,
            'corrupted_files': self.dataset_info['corrupted_files'],
            'categories': {}
        }

        # Per-category statistics
        for category, data in self.dataset_info['categories'].items():
            if not data:
                continue

            category_stats = {
                'file_count': data['valid_files'],
                'duration_seconds': data['total_duration'],
                'duration_minutes': data['total_duration'] / 60,
                'avg_duration': data['total_duration'] / data['valid_files'] if data['valid_files'] > 0 else 0,
                'sample_rates': {str(k): v for k, v in data['sample_rates'].items()},
                'formats': {str(k): v for k, v in data['formats'].items()},
                'quality_warnings_count': len(data['quality_warnings'])
            }

            stats['categories'][category] = category_stats

        return stats

    def save_manifest(self, output_path: Path):
        """
        Save dataset manifest to JSON

        Args:
            output_path: Path to output JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.dataset_info, f, indent=2)

        logger.info(f"Manifest saved to: {output_path}")


class DatasetSplitter:
    """Splits datasets into train/validation/test sets"""

    def __init__(self, dataset_info: Dict):
        """
        Initialize dataset splitter

        Args:
            dataset_info: Dataset information from scanner
        """
        self.dataset_info = dataset_info
        self.splits = {}

    def _find_npy_path(self, audio_path: Path, npy_dir: Path) -> Optional[str]:
        """
        Find corresponding .npy file for audio file

        Args:
            audio_path: Path to audio file
            npy_dir: Root directory containing .npy files

        Returns:
            Path to .npy file or None if not found
        """
        # Fast path: Try direct category-based matching first
        # Example: data/raw/positive/subdir/sample.wav -> data/raw/npy/positive/subdir/sample.npy
        try:
            # Get the category from audio path (e.g., "positive", "negative")
            parts = audio_path.parts
            if len(parts) >= 2:
                # Find category in path (positive, negative, hard_negative)
                for i, part in enumerate(parts):
                    if part in ['positive', 'negative', 'hard_negative']:
                        # Build NPY path with same relative structure after category
                        relative_parts = parts[i:]  # category/subdir/file.wav
                        npy_path = npy_dir.joinpath(*relative_parts).with_suffix('.npy')
                        if npy_path.exists():
                            return str(npy_path)
                        break
        except Exception as e:
            logger.debug(f"Fast path failed for {audio_path.name}: {e}")

        # Fallback: Try matching by filename only in category subdirectory
        # This avoids expensive recursive glob
        try:
            filename_stem = audio_path.stem
            # Try to find category
            for part in audio_path.parts:
                if part in ['positive', 'negative', 'hard_negative']:
                    category_npy_dir = npy_dir / part
                    if category_npy_dir.exists():
                        # Search only within category directory (much faster than full rglob)
                        for npy_file in category_npy_dir.rglob(f"{filename_stem}.npy"):
                            return str(npy_file)
                    break
        except Exception as e:
            logger.debug(f"Category search failed for {audio_path.name}: {e}")

        return None

    def split_datasets(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        stratify: bool = True,
        npy_source_dir: Path = Path("data/raw/npy"),
        npy_output_dir: Path = Path("data/npy")
    ) -> Dict:
        """
        Split datasets into train/val/test and organize NPY files

        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_seed: Random seed for reproducibility
            stratify: Use stratified splitting
            npy_source_dir: Source directory containing extracted .npy files
            npy_output_dir: Output directory for split .npy files (creates train/val/test subdirs)

        Returns:
            Dictionary with split information

        Note:
            NPY files will be automatically copied from npy_source_dir to npy_output_dir/{train,val,test}
        """
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

        logger.info(f"Splitting datasets: train={train_ratio}, val={val_ratio}, test={test_ratio}")

        splits = {
            'train': {'files': [], 'categories': defaultdict(int)},
            'val': {'files': [], 'categories': defaultdict(int)},
            'test': {'files': [], 'categories': defaultdict(int)}
        }

        # Split each category
        for category, data in self.dataset_info['categories'].items():
            if not data or not data['files']:
                logger.warning(f"No files in category: {category}")
                continue

            files = data['files']
            logger.info(f"Splitting {len(files)} files from {category}")

            # Create labels for stratification (all same category)
            labels = [category] * len(files)

            # First split: train vs (val + test)
            train_files, temp_files = train_test_split(
                files,
                test_size=(val_ratio + test_ratio),
                random_state=random_seed,
                stratify=labels if stratify and len(files) > 10 else None
            )

            # Second split: val vs test
            if len(temp_files) > 1:
                val_size = val_ratio / (val_ratio + test_ratio)
                val_files, test_files = train_test_split(
                    temp_files,
                    test_size=(1 - val_size),
                    random_state=random_seed
                )
            else:
                val_files = temp_files
                test_files = []

            # Add to splits with NPY path mapping
            # Only look for NPY files for training sample categories
            npy_categories = {'positive', 'negative', 'hard_negative'}
            should_find_npy = category in npy_categories

            for file_info in train_files:
                file_entry = {
                    'path': file_info['path'],
                    'category': category,
                    'duration': file_info.get('duration', 0.0),
                    'sample_rate': file_info.get('sample_rate', 16000)
                }
                # Find and map NPY file only for training categories
                if should_find_npy:
                    npy_path = self._find_npy_path(Path(file_info['path']), npy_source_dir)
                    if npy_path:
                        file_entry['npy_path'] = npy_path
                splits['train']['files'].append(file_entry)
                splits['train']['categories'][category] += 1

            for file_info in val_files:
                file_entry = {
                    'path': file_info['path'],
                    'category': category,
                    'duration': file_info.get('duration', 0.0),
                    'sample_rate': file_info.get('sample_rate', 16000)
                }
                if should_find_npy:
                    npy_path = self._find_npy_path(Path(file_info['path']), npy_source_dir)
                    if npy_path:
                        file_entry['npy_path'] = npy_path
                splits['val']['files'].append(file_entry)
                splits['val']['categories'][category] += 1

            for file_info in test_files:
                file_entry = {
                    'path': file_info['path'],
                    'category': category,
                    'duration': file_info.get('duration', 0.0),
                    'sample_rate': file_info.get('sample_rate', 16000)
                }
                if should_find_npy:
                    npy_path = self._find_npy_path(Path(file_info['path']), npy_source_dir)
                    if npy_path:
                        file_entry['npy_path'] = npy_path
                splits['test']['files'].append(file_entry)
                splits['test']['categories'][category] += 1

        # Convert defaultdicts to regular dicts
        for split_name in splits:
            splits[split_name]['categories'] = dict(splits[split_name]['categories'])
            splits[split_name]['total_files'] = len(splits[split_name]['files'])

        self.splits = splits

        logger.info(f"Split complete: train={len(splits['train']['files'])}, "
                   f"val={len(splits['val']['files'])}, test={len(splits['test']['files'])}")

        # Count how many files have NPY paths mapped
        npy_mapped_count = sum(
            1 for split_data in splits.values()
            for file_info in split_data['files']
            if file_info.get('npy_path')
        )
        logger.info(f"NPY paths mapped: {npy_mapped_count} out of {sum(len(s['files']) for s in splits.values())} files")

        # Automatically copy NPY files to split structure
        if npy_mapped_count > 0:
            logger.info(f"Copying NPY files from {npy_source_dir} to {npy_output_dir}...")
            npy_stats = self.copy_npy_files_to_splits(npy_output_dir, preserve_structure=True)

            total_copied = sum(s['copied'] for s in npy_stats.values())
            total_missing = sum(s['missing'] for s in npy_stats.values())
            logger.info(f"NPY organization complete: {total_copied} copied, {total_missing} missing")
        else:
            logger.warning(f"No NPY files found in {npy_source_dir}. Skipping NPY organization.")
            logger.warning("If you want to use precomputed features, run batch extraction first.")

        return splits

    def save_splits(self, output_dir: Path):
        """
        Save split manifests to JSON files

        Args:
            output_dir: Directory to save split files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in self.splits.items():
            output_path = output_dir / f"{split_name}.json"

            with open(output_path, 'w') as f:
                json.dump(split_data, f, indent=2)

            logger.info(f"Saved {split_name} split to: {output_path}")

        # Save summary
        summary_path = output_dir / "split_summary.json"
        summary = {
            'train': {
                'total_files': self.splits['train']['total_files'],
                'categories': self.splits['train']['categories']
            },
            'val': {
                'total_files': self.splits['val']['total_files'],
                'categories': self.splits['val']['categories']
            },
            'test': {
                'total_files': self.splits['test']['total_files'],
                'categories': self.splits['test']['categories']
            }
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved split summary to: {summary_path}")

    def copy_npy_files_to_splits(self, output_npy_dir: Path, preserve_structure: bool = True):
        """
        Physically copy NPY files into train/val/test directory structure

        This creates a mirrored directory structure where NPY files are organized
        by split and category, making it easier to load during training.

        Args:
            output_npy_dir: Root directory for split NPY files
            preserve_structure: If True, preserve category subdirectories

        Example output structure:
            output_npy_dir/
            ├── train/
            │   ├── positive/
            │   │   └── sample_001.npy
            │   └── negative/
            │       └── sample_002.npy
            ├── val/
            │   └── positive/
            │       └── sample_003.npy
            └── test/
                └── negative/
                    └── sample_004.npy
        """
        if not self.splits:
            raise ValueError("No splits created yet. Call split_datasets() first.")

        output_npy_dir = Path(output_npy_dir)
        logger.info(f"Copying NPY files to split directories in: {output_npy_dir}")

        # Categories that have NPY features extracted
        npy_categories = {'positive', 'negative', 'hard_negative'}

        stats = {
            'train': {'copied': 0, 'missing': 0},
            'val': {'copied': 0, 'missing': 0},
            'test': {'copied': 0, 'missing': 0}
        }

        for split_name, split_data in self.splits.items():
            split_dir = output_npy_dir / split_name

            # Filter files to only include categories that have NPY files
            files_with_npy = [f for f in split_data['files'] if f.get('category') in npy_categories]

            for file_info in tqdm(files_with_npy, desc=f"Copying {split_name} NPY files"):
                # Check if NPY path exists
                npy_path = file_info.get('npy_path')
                if not npy_path:
                    stats[split_name]['missing'] += 1
                    continue

                npy_source = Path(npy_path)
                if not npy_source.exists():
                    logger.warning(f"NPY file not found: {npy_source}")
                    stats[split_name]['missing'] += 1
                    continue

                # Determine destination path
                if preserve_structure:
                    # Preserve category subdirectories
                    category = file_info.get('category', 'unknown')
                    npy_dest = split_dir / category / npy_source.name
                else:
                    # Flat structure
                    npy_dest = split_dir / npy_source.name

                # Create parent directory
                npy_dest.parent.mkdir(parents=True, exist_ok=True)

                # Copy NPY file
                try:
                    shutil.copy2(npy_source, npy_dest)
                    stats[split_name]['copied'] += 1

                    # Update NPY path in split data to point to new location
                    file_info['npy_path'] = str(npy_dest)

                except Exception as e:
                    logger.error(f"Error copying {npy_source} to {npy_dest}: {e}")
                    stats[split_name]['missing'] += 1

        # Log statistics
        logger.info("NPY file copy statistics:")
        for split_name, split_stats in stats.items():
            logger.info(f"  {split_name}: {split_stats['copied']} copied, {split_stats['missing']} missing")

        total_copied = sum(s['copied'] for s in stats.values())
        total_missing = sum(s['missing'] for s in stats.values())
        logger.info(f"Total: {total_copied} copied, {total_missing} missing")

        return stats

    def get_split_statistics(self) -> Dict:
        """
        Get statistics about the splits

        Returns:
            Dictionary with split statistics
        """
        if not self.splits:
            raise ValueError("No splits created yet. Call split_datasets() first.")

        stats = {}

        for split_name, split_data in self.splits.items():
            total_files = split_data['total_files']
            categories = split_data['categories']

            stats[split_name] = {
                'total_files': total_files,
                'categories': categories,
                'percentage': 0.0
            }

        # Calculate percentages
        total_all = sum(s['total_files'] for s in stats.values())
        for split_name in stats:
            if total_all > 0:
                stats[split_name]['percentage'] = (stats[split_name]['total_files'] / total_all) * 100

        return stats


if __name__ == "__main__":
    # Test dataset scanner and splitter
    import sys

    print("Dataset Scanner and Splitter Test")
    print("=" * 60)

    # Example usage
    test_root = Path("data/raw")

    if test_root.exists():
        scanner = DatasetScanner(test_root)
        print(f"Scanner initialized for: {test_root}")

        # This would scan if data exists
        # results = scanner.scan_datasets()
        # print(f"Scan complete: {results['valid_files']} valid files found")

        print("Dataset scanner test complete")
    else:
        print(f"Test data directory not found: {test_root}")
        print("Scanner module loaded successfully")