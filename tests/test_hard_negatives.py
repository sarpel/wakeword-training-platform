import json
import tempfile
import os
from pathlib import Path
import pytest
from scripts.extract_hard_negatives import extract_false_positives

def test_extract_false_positives():
    # Create dummy log file (JSONL format)
    log_data = [
        {"path": "audio1.wav", "label": 0, "score": 0.95, "prediction": 1}, # FP, High conf
        {"path": "audio2.wav", "label": 1, "score": 0.99, "prediction": 1}, # TP
        {"path": "audio3.wav", "label": 0, "score": 0.10, "prediction": 0}, # TN
        {"path": "audio4.wav", "label": 0, "score": 0.85, "prediction": 1}, # FP, lower conf
    ]
    
    # We need to make sure the script directory exists or is in path if we import it
    # scripts/extract_hard_negatives.py
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        for entry in log_data:
            f.write(json.dumps(entry) + "\n")
        log_path = f.name
        
    try:
        # Extract FPs with score > 0.9
        # Assuming function returns list of dicts
        fps = extract_false_positives(log_path, threshold=0.9)
        
        assert len(fps) == 1
        assert fps[0]["path"] == "audio1.wav"
        
        # Test with lower threshold
        fps_low = extract_false_positives(log_path, threshold=0.8)
        assert len(fps_low) == 2
        
    finally:
        if os.path.exists(log_path):
            os.unlink(log_path)

