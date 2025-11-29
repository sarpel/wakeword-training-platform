import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

print("Verifying imports...")

try:
    from src.training import hpo
    print("✅ Successfully imported src.training.hpo")
except Exception as e:
    print(f"❌ Failed to import src.training.hpo: {e}")
    sys.exit(1)

try:
    from src.ui import panel_training
    print("✅ Successfully imported src.ui.panel_training")
except Exception as e:
    print(f"❌ Failed to import src.ui.panel_training: {e}")
    sys.exit(1)

print("Verifying HPO signature...")
import inspect
sig = inspect.signature(hpo.run_hpo)
if "log_callback" in sig.parameters:
    print("✅ run_hpo has log_callback parameter")
else:
    print("❌ run_hpo MISSING log_callback parameter")
    sys.exit(1)

print("Verifying UI functions...")
if hasattr(panel_training, "hpo_worker"):
    print("✅ panel_training has hpo_worker")
else:
    print("❌ panel_training MISSING hpo_worker")
    sys.exit(1)

print("All verification checks passed!")
