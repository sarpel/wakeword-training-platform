import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config.logger import setup_logging, get_data_logger

# Setup logging
setup_logging()
logger = get_data_logger("test_enhanced")

print("Testing Dataset Logging...")
# Simulate Dataset Panel logging
report_text = """
============================================================
DATASET SPLIT SUMMARY
============================================================

TRAIN:
  Total Files: 100
  Percentage: 70.0%
  Categories: 2

VAL:
  Total Files: 20
  Percentage: 15.0%
  Categories: 2

TEST:
  Total Files: 20
  Percentage: 15.0%
  Categories: 2

✅ Splits saved to: data/splits
============================================================
"""
logger.info("Dataset split complete")
logger.info(report_text)


print("Testing Trainer Logging...")
# Simulate Trainer logging
epoch = 0
train_loss = 0.5
train_acc = 0.8
val_loss = 0.4
val_acc = 0.85
val_f1 = 0.82
val_fpr = 0.05
val_fnr = 0.1
current_lr = 0.001

logger.info(f"Epoch {epoch+1}/10")
logger.info(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
logger.info(
    f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}, "
    f"F1={val_f1:.4f}, FPR={val_fpr:.4f}, FNR={val_fnr:.4f}"
)
logger.info(f"  LR: {current_lr:.6f}")
logger.info(f"  ✅ New best model (improvement detected)")

print("Verifying log file content...")
log_dir = Path("logs")
log_files = list(log_dir.glob("wakeword_training_*.log"))

if not log_files:
    print("❌ No log files found")
    sys.exit(1)

latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
print(f"Checking latest log file: {latest_log}")

content = latest_log.read_text(encoding="utf-8")

checks = [
    "DATASET SPLIT SUMMARY",
    "TRAIN:",
    "Epoch 1/10",
    "Train: Loss=0.5000",
    "Val:   Loss=0.4000",
    "New best model"
]

all_passed = True
for check in checks:
    if check in content:
        print(f"✅ Found: {check}")
    else:
        print(f"❌ Missing: {check}")
        all_passed = False

if all_passed:
    print("All verification checks passed!")
else:
    print("❌ Some checks failed")
    sys.exit(1)
