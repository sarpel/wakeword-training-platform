"""
Project Path Configuration
Centralizes all path definitions to avoid hardcoding and ensure consistency.
"""
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass
class ProjectPaths:
    """Centralized project path definitions"""
    
    # Root directories
    ROOT: Path = Path(os.getcwd())
    DATA: Path = ROOT / "data"
    SRC: Path = ROOT / "src"
    MODELS: Path = ROOT / "models"
    CONFIGS: Path = ROOT / "configs"
    
    # Data subdirectories
    RAW_DATA: Path = DATA / "raw"
    PROCESSED_DATA: Path = DATA / "processed"
    SPLITS: Path = DATA / "splits"
    NPY_FEATURES: Path = DATA / "npy"
    CMVN_STATS: Path = DATA / "cmvn_stats.json"
    
    # Model subdirectories
    CHECKPOINTS: Path = MODELS / "checkpoints"
    EXPORTS: Path = MODELS / "exports"
    
    # Raw data categories
    BACKGROUND_NOISE: Path = RAW_DATA / "background"
    RIRS: Path = RAW_DATA / "rirs"
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all critical directories exist"""
        for path in [
            cls.DATA, cls.RAW_DATA, cls.SPLITS, cls.NPY_FEATURES,
            cls.MODELS, cls.CHECKPOINTS, cls.EXPORTS, cls.CONFIGS
        ]:
            path.mkdir(parents=True, exist_ok=True)

# Global instance
paths = ProjectPaths()
