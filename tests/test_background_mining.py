import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.evaluation.background_miner import BackgroundMiner


def test_background_miner_persistence(tmp_path):
    """Verify that mining sessions are saved and loaded correctly."""
    sessions_file = tmp_path / "sessions.json"

    mock_evaluator = MagicMock()
    mock_evaluator.sample_rate = 16000

    miner = BackgroundMiner(mock_evaluator, sessions_path=str(sessions_file))

    # Simulate a session
    file_id = "test_file.wav"
    miner.sessions[file_id] = {"last_processed_sec": 10.5, "found_count": 2}
    miner._save_sessions()

    # Load again
    miner2 = BackgroundMiner(mock_evaluator, sessions_path=str(sessions_file))
    assert miner2.sessions[file_id]["last_processed_sec"] == 10.5
    assert miner2.sessions[file_id]["found_count"] == 2


def test_background_miner_processing(tmp_path):
    """Verify the processing loop logic."""
    sessions_file = tmp_path / "sessions.json"

    # Create a dummy audio file (2 seconds)
    import soundfile as sf

    audio_path = tmp_path / "long_background.wav"
    sr = 16000
    audio = np.random.randn(sr * 2)
    sf.write(audio_path, audio, sr)

    mock_evaluator = MagicMock()
    mock_evaluator.sample_rate = sr
    # Always return high confidence to trigger "found"
    mock_evaluator.evaluate_audio.return_value = (0.9, True)

    miner = BackgroundMiner(mock_evaluator, sessions_path=str(sessions_file))

    # Process 1.0s with 0.5s hop -> should find several windows
    result = miner.process_file(audio_path, window_duration_s=1.0, hop_duration_s=0.5, resume=False)

    assert result["found"] > 0
    assert result["total_sec"] == 2.0
    assert miner.sessions[str(audio_path.absolute())]["status"] == "completed"
