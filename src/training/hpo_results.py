
from typing import Dict, Any, Union, List
from dataclasses import dataclass, field

@dataclass
class HPOResult:
    """
    Standardized result from an HPO trial or study.
    """
    study_name: str
    best_value: float
    best_params: Dict[str, Union[int, float, str]]
    n_trials: int
    duration: float
    
    # Optional: Detailed history if needed for UI plots
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "study_name": self.study_name,
            "best_value": self.best_value,
            "best_params": self.best_params,
            "n_trials": self.n_trials,
            "duration": self.duration
        }
