"""
Hard Negative Mining for Wakeword Detection
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import structlog

from src.evaluation.types import EvaluationResult

logger = structlog.get_logger(__name__)

class HardNegativeMiner:
    """
    Identifies and manages 'hard negative' samples (false positives)
    from evaluation results for retraining.
    """
    
    def __init__(self, queue_path: str = "logs/mining_queue.json"):
        self.queue_path = Path(queue_path)
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        self.queue = self._load_queue()
        
    def _load_queue(self) -> List[Dict[str, Any]]:
        if self.queue_path.exists():
            with open(self.queue_path, "r") as f:
                return json.load(f)
        return []
        
    def _save_queue(self):
        with open(self.queue_path, "w") as f:
            json.dump(self.queue, f, indent=2)
            
    def mine_from_results(self, results: List[EvaluationResult], confidence_threshold: float = 0.7) -> int:
        """
        Identify False Positives from results and add to queue.
        
        Args:
            results: List of EvaluationResult objects.
            confidence_threshold: Only mine FP with confidence above this.
            
        Returns:
            Number of new samples added.
        """
        mined_count = 0
        existing_paths = {item["full_path"] for item in self.queue}
        
        for res in results:
            # False Positive: predicted Positive, label is Negative
            if res.prediction == "Positive" and res.label == 0:
                if res.confidence >= confidence_threshold:
                    if res.full_path and res.full_path not in existing_paths:
                        self.queue.append({
                            "filename": res.filename,
                            "full_path": res.full_path,
                            "confidence": float(res.confidence),
                            "status": "pending", # pending, confirmed, discarded
                            "timestamp": os.path.getmtime(res.full_path) if os.path.exists(res.full_path) else 0
                        })
                        mined_count += 1
                        
        if mined_count > 0:
            self._save_queue()
            logger.info(f"Mined {mined_count} new hard negatives into {self.queue_path}")
            
        return mined_count
        
    def get_pending(self) -> List[Dict[str, Any]]:
        return [item for item in self.queue if item["status"] == "pending"]
        
    def update_status(self, full_path: str, status: str):
        for item in self.queue:
            if item["full_path"] == full_path:
                item["status"] = status
                break
        self._save_queue()
        
    def inject_to_dataset(self, target_dir: str = "data/mined_negatives"):
        """
        Copy confirmed hard negatives to a target directory for retraining.
        """
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        injected_count = 0
        for item in self.queue:
            if item["status"] == "confirmed":
                src = Path(item["full_path"])
                if src.exists():
                    dst = target_path / src.name
                    if not dst.exists():
                        shutil.copy(src, dst)
                        injected_count += 1
                        
        logger.info(f"Injected {injected_count} confirmed negatives into {target_dir}")
        return injected_count
