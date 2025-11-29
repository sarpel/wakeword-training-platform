"""
Data Preprocessing and Cleaning
Includes VAD filtering and other offline processing steps.
"""
import json
from pathlib import Path
from typing import List, Dict, Optional
import shutil
import structlog
from tqdm import tqdm
import torch
import torchaudio

from src.data.vad import EnergyVAD

logger = structlog.get_logger(__name__)

class VADFilter:
    """
    Filter dataset using Voice Activity Detection.
    Removes samples that contain only silence or noise.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000, 
                 energy_threshold: float = 0.05):
        self.vad = EnergyVAD(sample_rate=sample_rate, energy_threshold=energy_threshold)
        self.sample_rate = sample_rate

    def process_dataset(self, 
                        manifest_path: Path, 
                        output_path: Optional[Path] = None, 
                        min_speech_duration: float = 0.1) -> Path:
        """
        Filter a dataset manifest, removing non-speech files.
        
        Args:
            manifest_path: Path to input JSON manifest
            output_path: Path to save filtered JSON manifest (default: manifest_path_cleaned.json)
            min_speech_duration: Minimum duration of speech to keep file (not fully used by EnergyVAD yet, but reserved)
            
        Returns:
            Path to new manifest
        """
        manifest_path = Path(manifest_path)
        if output_path is None:
            output_path = manifest_path.parent / f"{manifest_path.stem}_cleaned.json"
            
        with open(manifest_path, "r") as f:
            data = json.load(f)
            
        files = data["files"]
        kept_files = []
        rejected_count = 0
        
        logger.info(f"VAD Filtering: Processing {len(files)} files from {manifest_path}")
        
        for item in tqdm(files, desc="VAD Filtering"):
            file_path = Path(item["path"])
            category = item["category"]
            
            # Skip VAD for background noise or explicit negatives if desired?
            # Usually we want VAD on positives to ensure they have speech.
            # Negatives might be silence/noise, so maybe we keep them?
            # For now, we apply to ALL, assuming "negative" usually means "speech that isn't wakeword".
            # BUT if negative category includes background noise, VAD might filter it out.
            # Let's safeguard: If category is "background", keep it.
            if category == "background":
                kept_files.append(item)
                continue

            try:
                # Load audio
                waveform, sr = torchaudio.load(file_path)
                if sr != self.sample_rate:
                    waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
                
                # Check VAD
                if self.vad.is_speech(waveform):
                    kept_files.append(item)
                else:
                    rejected_count += 1
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                rejected_count += 1
                
        # Save new manifest
        new_data = data.copy()
        new_data["files"] = kept_files
        
        with open(output_path, "w") as f:
            json.dump(new_data, f, indent=2)
            
        logger.info(f"VAD Filter Complete. Kept {len(kept_files)}/{len(files)}. Rejected {rejected_count}.")
        logger.info(f"Cleaned manifest saved to {output_path}")
        
        return output_path

def clean_dataset_split(data_root: str, split: str = "train"):
    """
    Convenience function to clean a specific split.
    """
    root = Path(data_root)
    manifest = root / "splits" / f"{split}.json"
    
    if not manifest.exists():
        logger.error(f"Manifest not found: {manifest}")
        return
        
    vad_filter = VADFilter()
    vad_filter.process_dataset(manifest)

if __name__ == "__main__":
    # Test
    print("VAD Filter Test")
    # Mock functionality or run on dummy file if needed
