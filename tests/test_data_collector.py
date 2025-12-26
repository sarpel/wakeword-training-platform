import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.data_collector import FalsePositiveCollector


def test_collector_initialization():
    """Test initializing the collector."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = FalsePositiveCollector(output_dir=tmpdir)
        assert Path(tmpdir).exists()


def test_add_sample():
    """Test adding a false positive sample."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = FalsePositiveCollector(output_dir=tmpdir)

        audio = np.random.randn(16000).astype(np.float32)
        metadata = {"filename": "test.wav", "confidence": 0.8}

        collector.add_sample(audio, metadata)

        # Check if files were created
        saved_files = list(Path(tmpdir).glob("*.wav"))
        assert len(saved_files) == 1

        # Check index
        index_file = Path(tmpdir) / "index.json"
        assert index_file.exists()

        with open(index_file) as f:
            index = json.load(f)
            assert len(index) == 1
            assert index[0]["metadata"]["filename"] == "test.wav"


def test_limit_samples():
    """Test that the collector respects the max samples limit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = FalsePositiveCollector(output_dir=tmpdir, max_samples=2)

        audio = np.zeros(16000, dtype=np.float32)

        collector.add_sample(audio, {"id": 1})
        collector.add_sample(audio, {"id": 2})
        collector.add_sample(audio, {"id": 3})

        index_file = Path(tmpdir) / "index.json"
        with open(index_file) as f:
            index = json.load(f)
            assert len(index) == 2
