from dataclasses import dataclass
import numpy as np

@dataclass
class EvaluationResult:
    """Single file evaluation result"""

    filename: str
    prediction: str  # "Positive" or "Negative"
    confidence: float
    latency_ms: float
    logits: np.ndarray
