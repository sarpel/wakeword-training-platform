"""
File Cache for Dataset Operations
Caches file metadata to speed up subsequent scans
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

logger = structlog.get_logger(__name__)


class FileCache:
    """Cache for file metadata to avoid redundant validation"""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        """
        Initialize file cache

        Args:
            cache_dir: Directory to store cache files (default: data/cache)
        """
        if cache_dir is None:
            cache_dir = Path("data/cache")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, Any] = {}
        self.cache_file = self.cache_dir / "file_metadata_cache.json"

        # Load existing cache
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded cache with {len(self.cache)} entries")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.cache = {}
        else:
            self.cache = {}

    def _save_cache(self) -> None:
        """Save cache to disk"""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f, indent=2)
            logger.debug(f"Saved cache with {len(self.cache)} entries")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _get_file_key(self, file_path: Path) -> str:
        """
        Get unique key for file based on path and modification time

        Args:
            file_path: Path to file

        Returns:
            Cache key string
        """
        file_path = Path(file_path)
        stat = file_path.stat()

        # Create key from path and mtime
        key_data = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()

        return key_hash

    def get(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Get cached metadata for file

        Args:
            file_path: Path to file

        Returns:
            Cached metadata or None if not found/invalid
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                return None

            key = self._get_file_key(file_path)

            if key in self.cache:
                # Mypy: Explicitly type the cached data to avoid Any return
                cached_data: Dict[str, Any] = self.cache[key]
                logger.debug(f"Cache hit for {file_path.name}")
                return cached_data

            return None

        except Exception as e:
            logger.debug(f"Cache get error for {file_path}: {e}")
            return None

    def set(self, file_path: Path, metadata: Dict[str, Any]) -> None:
        """
        Cache metadata for file

        Args:
            file_path: Path to file
            metadata: Metadata to cache
        """
        try:
            file_path = Path(file_path)
            key = self._get_file_key(file_path)

            # Add cache timestamp
            metadata["_cached_at"] = datetime.now().isoformat()

            self.cache[key] = metadata
            logger.debug(f"Cached metadata for {file_path.name}")

        except Exception as e:
            logger.debug(f"Cache set error for {file_path}: {e}")

    def save(self) -> None:
        """Save cache to disk"""
        self._save_cache()

    def clear(self) -> None:
        """Clear all cache"""
        self.cache = {}
        self._save_cache()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dictionary with cache stats
        """
        return {
            "total_entries": len(self.cache),
            "cache_file": str(self.cache_file),
            "cache_exists": self.cache_file.exists(),
        }


if __name__ == "__main__":
    # Test file cache
    print("File Cache Test")
    print("=" * 60)

    cache = FileCache()
    print(f"Cache initialized: {cache.cache_file}")
    print(f"Cache stats: {cache.get_stats()}")

    print("\nFile Cache test complete")
