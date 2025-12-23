"""
Data Preprocessing and Cleaning
Includes VAD filtering and other offline processing steps.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import structlog
import torchaudio
from tqdm import tqdm

from src.data.vad import EnergyVAD

logger = structlog.get_logger(__name__)


class VADFilter:
    """
    Filter dataset using Voice Activity Detection.
    Removes samples that contain only silence or noise.
    """

    def __init__(self, sample_rate: int = 16000, energy_threshold: float = 0.05):
        self.vad = EnergyVAD(sample_rate=sample_rate, energy_threshold=energy_threshold)
        self.sample_rate = sample_rate

    def process_dataset(
        self, manifest_path: Path, output_path: Optional[Path] = None, min_speech_duration: float = 0.1
    ) -> Path:
        """
        Filter a dataset manifest, removing non-speech files.

        Args:
            manifest_path: Path to input JSON manifest
            output_path: Path to save filtered JSON manifest (default: manifest_path_cleaned.json)
            min_speech_duration: Minimum duration of speech to keep file (not fully used by EnergyVAD yet, but reserved)

        Returns:
            Path to new manifest
        """
        manifest_path = Path(manifest_path)
        if output_path is None:
            output_path = manifest_path.parent / f"{manifest_path.stem}_cleaned.json"

        with open(manifest_path, "r") as f:
            data = json.load(f)

        rejected_count = 0
        total_files = 0

        # Handle both flat list (legacy) and nested categories (new scanner)
        if "files" in data:
            # Legacy flat structure
            files = data["files"]
            kept_files = []
            total_files = len(files)

            logger.info(f"VAD Filtering: Processing {len(files)} files from {manifest_path}")

            for item in tqdm(files, desc="VAD Filtering"):
                if self._process_item(item):
                    kept_files.append(item)
                else:
                    rejected_count += 1

            data["files"] = kept_files

        elif "categories" in data:
            # New nested structure
            logger.info(f"VAD Filtering: Processing categories in {manifest_path}")

            for category_name, category_data in data["categories"].items():
                files = category_data.get("files", [])
                kept_files = []
                total_files += len(files)

                # Skip background noise and RIRs from VAD
                # Usually background noise is just noise, VAD might reject it all.
                # But we want to keep it as "background" class.
                if category_name in ["background", "rir", "rirs"]:
                    logger.info(f"Skipping VAD for category '{category_name}' (keeping all {len(files)} files)")
                    kept_files = files
                else:
                    for item in tqdm(files, desc=f"Filtering {category_name}"):
                        if self._process_item(item):
                            kept_files.append(item)
                        else:
                            rejected_count += 1

                # Update category files
                data["categories"][category_name]["files"] = kept_files
                # Update category stats if present
                if "total_files" in data["categories"][category_name]:
                    data["categories"][category_name]["total_files"] = len(kept_files)

        else:
            logger.error(f"Unknown manifest format in {manifest_path}")
            return output_path

        # Update global stats if present
        if "total_files" in data:
            data["total_files"] = total_files - rejected_count

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"VAD Filter Complete. Rejected {rejected_count} files.")
        logger.info(f"Cleaned manifest saved to {output_path}")

        return output_path

    def _process_item(self, item: Dict) -> bool:
        """Process a single file item. Returns True to keep, False to reject."""
        file_path = Path(item["path"])
        category = item.get("category", "")

        # Double check category skip if flat list used
        if category in ["background", "rir", "rirs"]:
            return True

        try:
            # Load audio
            if not file_path.exists():
                return False

            waveform, sr = torchaudio.load(file_path)
            if sr != self.sample_rate:
                waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

            # Check VAD
            return self.vad.is_speech(waveform)
        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")
            return False


def clean_dataset_split(data_root: str, split: str = "train") -> None:
    """
    Convenience function to clean a specific split.
    """
    root = Path(data_root)
    manifest = root / "splits" / f"{split}.json"

    if not manifest.exists():
        logger.error(f"Manifest not found: {manifest}")
        return

    vad_filter = VADFilter()
    vad_filter.process_dataset(manifest)


if __name__ == "__main__":
    # Test
    print("VAD Filter Test")
    # Mock functionality or run on dummy file if needed
