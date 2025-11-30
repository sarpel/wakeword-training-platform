import sys
from pathlib import Path
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config.logger import setup_logging, get_data_logger

# Clean up previous log file if exists
log_file = Path("logs/wakeword_training.log")
if log_file.exists():
    log_file.unlink()

print("Setting up logging...")
setup_logging()
logger = get_data_logger("test")

print("Writing test log message...")
test_message = "VERIFICATION_TEST_MESSAGE_12345"
logger.info(test_message)

print("Verifying log file content...")
if not log_file.exists():
    print(f"❌ Log file not created at {log_file}")
    sys.exit(1)

content = log_file.read_text(encoding="utf-8")
if test_message in content:
    print("✅ Log message found in file")
else:
    print(f"❌ Log message NOT found in file. Content:\n{content}")
    sys.exit(1)

print("All verification checks passed!")
