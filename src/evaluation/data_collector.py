"""
Utilities for collecting and managing analysis data (e.g. False Positives).
"""

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
import structlog

logger = structlog.get_logger(__name__)


class FalsePositiveCollector:
    """
    Collects false positive audio samples and metadata for inspection.
    """

    def __init__(self, output_dir: str = "data/analysis/false_positives", max_samples: int = 100):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples
        self.index_file = self.output_dir / "index.json"
        self._load_index()

    def _load_index(self) -> None:
        if self.index_file.exists():
            with open(self.index_file, "r") as f:
                data = json.load(f)
                self.index = list(data) if isinstance(data, list) else []
        else:
            self.index = []

    def _save_index(self) -> None:
        with open(self.index_file, "w") as f:
            json.dump(self.index, f, indent=2)

    def add_sample(self, audio: np.ndarray, metadata: Dict[str, Any], sample_rate: int = 16000) -> None:
        """
        Add a new sample to the collection.

        Args:
            audio: Audio waveform array
            metadata: Arbitrary metadata (confidence, model info, etc.)
            sample_rate: Sample rate for saving audio
        """
        if len(self.index) >= self.max_samples:
            return

        sample_id = str(uuid.uuid4())
        filename = f"{sample_id}.wav"
        file_path = self.output_dir / filename

        # Save audio
        sf.write(str(file_path), audio, sample_rate)

        # Update index
        entry = {
            "id": sample_id,
            "audio_path": str(filename),
            "metadata": metadata,
            "timestamp": str(uuid.uuid1().time),  # Simple timestamp proxy
        }
        self.index.append(entry)
        self._save_index()

    def get_samples(self) -> List[Dict[str, Any]]:
        """Return the list of collected samples."""
        return self.index

    def clear(self) -> None:
        """Clear all collected samples."""
        # Delete files
        for entry in self.index:
            path = self.output_dir / entry["audio_path"]
            if path.exists():
                path.unlink()

        # Reset index
        self.index = []
        self._save_index()
