import sys
import time
from pathlib import Path
import glob

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config.logger import setup_logging, get_data_logger

print("Setting up logging...")
setup_logging()
logger = get_data_logger("test_timestamp")

print("Writing test log message...")
test_message = f"TIMESTAMP_TEST_{time.time()}"
logger.info(test_message)

print("Verifying log file creation...")
log_dir = Path("logs")
log_files = list(log_dir.glob("wakeword_training_*.log"))

if not log_files:
    print("❌ No timestamped log files found in logs/")
    sys.exit(1)

# Find the most recent file
latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
print(f"Checking latest log file: {latest_log}")

content = latest_log.read_text(encoding="utf-8")
if test_message in content:
    print("✅ Log message found in timestamped file")
else:
    print(f"❌ Log message NOT found in {latest_log}. Content:\n{content}")
    sys.exit(1)

print("All verification checks passed!")
