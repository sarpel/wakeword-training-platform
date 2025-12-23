from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


class StageBase(ABC):
    """Base class for all inference stages (e.g., Sentry, Judge)"""

    @abstractmethod
    def predict(self, audio: np.ndarray) -> dict:
        """Run prediction on an audio segment"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the stage"""
        pass


class InferenceEngine(ABC):
    """Base class for the orchestration engine"""

    @abstractmethod
    def add_stage(self, stage: StageBase) -> None:
        """Add a stage to the pipeline"""
        pass

    @abstractmethod
    def run(self, audio: np.ndarray) -> list:
        """Execute the full pipeline"""
        pass


@dataclass
class EvaluationResult:
    """Single file evaluation result"""

    filename: str
    prediction: str  # "Positive" or "Negative"
    confidence: float
    latency_ms: float
    logits: np.ndarray
    label: Optional[int] = None
    raw_audio: Optional[np.ndarray] = None
    full_path: Optional[str] = None
