
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

print("Checking imports...")

try:
    print("Importing WakewordDataset...")
    from src.data.dataset import WakewordDataset
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")

try:
    print("Importing CMVN...")
    from src.data.cmvn import CMVN
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")

try:
    print("Importing Architectures...")
    from src.models.architectures import LSTMWakeword, GRUWakeword
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")

try:
    print("Importing BalancedSampler...")
    from src.data.balanced_sampler import create_balanced_sampler_from_dataset
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")

try:
    print("Importing Losses...")
    from src.models.losses import LabelSmoothingCrossEntropy
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")

try:
    print("Importing AudioProcessor...")
    from src.data.processor import AudioProcessor
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")

try:
    print("Importing CUDA Utils...")
    from src.config.cuda_utils import enforce_cuda
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")

print("Done.")
