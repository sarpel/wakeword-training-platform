import pytest
import os
import json
from pathlib import Path
from src.evaluation.mining import HardNegativeMiner
from src.evaluation.types import EvaluationResult
import numpy as np

def test_mine_hard_negatives(tmp_path):
    queue_file = tmp_path / "queue.json"
    miner = HardNegativeMiner(queue_path=str(queue_file))
    
    # Create fake audio file
    fake_audio = tmp_path / "fake.wav"
    fake_audio.write_text("dummy")
    
    results = [
        EvaluationResult(
            filename="fake.wav",
            prediction="Positive",
            confidence=0.9,
            latency_ms=1.0,
            logits=np.array([0.1, 0.9]),
            label=0, # False Positive!
            full_path=str(fake_audio)
        ),
        EvaluationResult(
            filename="true_pos.wav",
            prediction="Positive",
            confidence=0.9,
            latency_ms=1.0,
            logits=np.array([0.1, 0.9]),
            label=1, # True Positive
            full_path="true_pos.wav"
        )
    ]
    
    count = miner.mine_from_results(results)
    assert count == 1
    assert len(miner.queue) == 1
    assert miner.queue[0]["status"] == "pending"

def test_update_and_inject(tmp_path):
    queue_file = tmp_path / "queue.json"
    miner = HardNegativeMiner(queue_path=str(queue_file))
    
    fake_audio = tmp_path / "fake.wav"
    fake_audio.write_text("dummy")
    
    miner.queue = [{
        "filename": "fake.wav",
        "full_path": str(fake_audio),
        "confidence": 0.9,
        "status": "pending"
    }]
    
    miner.update_status(str(fake_audio), "confirmed")
    assert miner.queue[0]["status"] == "confirmed"
    
    inject_dir = tmp_path / "mined"
    count = miner.inject_to_dataset(target_dir=str(inject_dir))
    assert count == 1
    assert (inject_dir / "fake.wav").exists()
