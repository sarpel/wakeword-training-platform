
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
import torch

from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.mining import HardNegativeMiner
from src.evaluation.types import EvaluationResult

logger = structlog.get_logger(__name__)

class BackgroundMiner:
    """
    Analyzes long background audio files to identify false positives.
    Supports session persistence to resume processing.
    """

    def __init__(
        self,
        evaluator: ModelEvaluator,
        miner: Optional[HardNegativeMiner] = None,
        sessions_path: str = "logs/mining_sessions.json",
    ):
        self.evaluator = evaluator
        self.miner = miner or HardNegativeMiner()
        self.sessions_path = Path(sessions_path)
        self.sessions_path.parent.mkdir(parents=True, exist_ok=True)
        self.sessions = self._load_sessions()

    def _load_sessions(self) -> Dict[str, Any]:
        if self.sessions_path.exists():
            try:
                with open(self.sessions_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load sessions: {e}")
        return {}

    def _save_sessions(self) -> None:
        with open(self.sessions_path, "w", encoding="utf-8") as f:
            json.dump(self.sessions, f, indent=2)

    def process_file(
        self,
        file_path: Path,
        window_duration_s: float = 1.5,
        hop_duration_s: float = 0.5,
        threshold: float = 0.4,
        resume: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Process a long audio file in windows.
        """
        file_path = Path(file_path)
        file_id = str(file_path.absolute())
        
        # Load audio
        import librosa
        audio, sr = librosa.load(file_path, sr=self.evaluator.sample_rate)
        duration = len(audio) / sr
        
        start_sec = 0.0
        if resume and file_id in self.sessions:
            start_sec = self.sessions[file_id].get("last_processed_sec", 0.0)
            logger.info(f"Resuming {file_path.name} from {start_sec:.2f}s")

        window_samples = int(window_duration_s * sr)
        hop_samples = int(hop_duration_s * sr)
        
        current_sample = int(start_sec * sr)
        total_samples = len(audio)
        
        found_count = 0
        results = []

        while current_sample + window_samples <= total_samples:
            window = audio[current_sample : current_sample + window_samples]
            
            # Predict
            # We wrap window in EvaluationResult logic or similar
            # ModelEvaluator.evaluate_audio returns (confidence, is_positive)
            confidence, is_positive = self.evaluator.evaluate_audio(window, threshold=threshold)
            
            if confidence >= threshold:
                # Potential False Positive found
                # Create a result object for the miner
                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                # Save chunk to temp file for miner to pick up
                temp_dir = Path("data/cache/mining_chunks")
                temp_dir.mkdir(parents=True, exist_ok=True)
                chunk_name = f"chunk_{file_path.stem}_{int(current_sample/sr)}s_{timestamp_str}.wav"
                chunk_path = temp_dir / chunk_name
                
                import soundfile as sf
                sf.write(chunk_path, window, sr)
                
                res = EvaluationResult(
                    filename=chunk_name,
                    prediction="Positive",
                    confidence=float(confidence),
                    label=0, # It's background, so label is definitely Negative
                    full_path=str(chunk_path.absolute()),
                    logits=None, # Not needed for miner
                    raw_audio=window,
                    latency_ms=0.0
                )
                results.append(res)
                found_count += 1

            current_sample += hop_samples
            
            # Update session frequently
            processed_sec = current_sample / sr
            self.sessions[file_id] = {
                "last_processed_sec": processed_sec,
                "total_duration": duration,
                "found_count": self.sessions.get(file_id, {}).get("found_count", 0) + (1 if confidence >= threshold else 0)
            }
            if found_count % 10 == 0:
                self._save_sessions()
                
            if progress_callback:
                progress_callback(processed_sec / duration, f"Processing {file_path.name}: {processed_sec:.1f}s / {duration:.1f}s")

        # Finalize
        if results:
            self.miner.mine_from_results(results)
            
        self.sessions[file_id]["status"] = "completed" if current_sample + window_samples >= total_samples else "paused"
        self._save_sessions()
        
        return {
            "file": file_path.name,
            "found": found_count,
            "processed_sec": current_sample / sr,
            "total_sec": duration
        }
