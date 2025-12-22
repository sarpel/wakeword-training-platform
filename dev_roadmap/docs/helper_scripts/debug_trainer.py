import sys
import os
import inspect
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    from src.training.trainer import Trainer
    print(f"Trainer file: {inspect.getfile(Trainer)}")
    print(f"Trainer init signature: {inspect.signature(Trainer.__init__)}")
except Exception as e:
    print(f"Error importing Trainer: {e}")

try:
    from src.training.distillation_trainer import DistillationTrainer
    print(f"DistillationTrainer file: {inspect.getfile(DistillationTrainer)}")
    print(f"DistillationTrainer init signature: {inspect.signature(DistillationTrainer.__init__)}")
except Exception as e:
    print(f"Error importing DistillationTrainer: {e}")
