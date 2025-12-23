"""
Hard Negative Mining for Wakeword Detection
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

import structlog

from src.evaluation.types import EvaluationResult

logger = structlog.get_logger(__name__)


class HardNegativeMiner:
    """
    Identifies and manages 'hard negative' samples (false positives) from evaluation results.
    
    A 'hard negative' is a sound clip that the model incorrectly predicted as a wakeword
    (False Positive), but which is actually NOT a wakeword (Negative label). These samples
    help train the model to distinguish the wakeword from similar-sounding words or sounds.
    """

    def __init__(self, queue_path: str = "logs/mining_queue.json"):
        self.queue_path = Path(queue_path)
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        self.queue = self._load_queue()

    def _load_queue(self) -> List[Dict[str, Any]]:
        if self.queue_path.exists():
            with open(self.queue_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return list(data) if isinstance(data, list) else []
        return []

    def _save_queue(self) -> None:
        with open(self.queue_path, "w", encoding="utf-8") as f:
            json.dump(self.queue, f, indent=2)

    def mine_from_results(self, results: List[EvaluationResult], confidence_threshold: float = 0.7) -> int:
        """
        Identify False Positives (model says "wakeword" but label is "not wakeword") from results.

        Args:
            results: List of EvaluationResult objects.
            confidence_threshold: Only find FP with confidence above this.

        Returns:
            Number of new samples added to verification queue.
        """
        mined_count = 0
        existing_paths = {item["full_path"] for item in self.queue}

        for res in results:
            # False Positive: model predicted "Positive" (wakeword) but actual label is Negative
            if res.prediction == "Positive" and res.label == 0:
                if res.confidence >= confidence_threshold:
                    if res.full_path and res.full_path not in existing_paths:
                        self.queue.append(
                            {
                                "filename": res.filename,
                                "full_path": res.full_path,
                                "confidence": float(res.confidence),
                                "status": "pending",  # pending, confirmed, discarded
                                "timestamp": os.path.getmtime(res.full_path) if os.path.exists(res.full_path) else 0,
                            }
                        )
                        mined_count += 1

        if mined_count > 0:
            self._save_queue()
            logger.info(f"Found {mined_count} false positives (model detected as wakeword but are NOT). Saved to {self.queue_path}")

        return mined_count

    def get_pending(self) -> List[Dict[str, Any]]:
        """
        Get all samples awaiting user verification.
        """
        return [item for item in self.queue if item["status"] == "pending"]

    def update_status(self, full_path: str, status: str) -> None:
        """
        Update verification status of a sample.
        
        Status values:
            - "pending": Awaiting user review
            - "confirmed": Verified as NOT wakeword (will be added to training)
            - "discarded": Rejected, will NOT be added to training
        """
        for item in self.queue:
            if item["full_path"] == full_path:
                item["status"] = status
                break
        self._save_queue()

    def inject_to_dataset(self, target_dir: str = "data/mined_negatives") -> int:
        """
        Copy verified hard negatives (confirmed "not wakeword" samples) to target directory for retraining.
        These samples help model learn the difference between wakeword and similar-sounding words.
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
