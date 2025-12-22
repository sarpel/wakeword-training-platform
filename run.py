"""
Quick Launcher for Wakeword Training Platform
Production-Ready Wakeword Detection Training System v2.0.0
"""
import sys
import os
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_requirements():
    """Check if essential requirements are installed"""
    missing = []

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import gradio
    except ImportError:
        missing.append("gradio")

    try:
        import librosa
    except ImportError:
        missing.append("librosa")

    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print(f"   Please install: pip install -r requirements.txt")
        return False

    return True

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return True, f"{gpu_name} ({gpu_memory:.1f} GB)"
        else:
            return False, "No CUDA GPU detected"
    except Exception as e:
        return False, f"Error checking CUDA: {e}"

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        "data/raw/positive",
        "data/raw/negative",
        "data/splits",
        "models/checkpoints",
        "exports",
    ]

    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)

def print_banner():
    """Print welcome banner with system info"""
    cuda_available, cuda_info = check_cuda()

    banner = """
+----------------------------------------------------------------------+
|                                                                      |
|        Wakeword Training Platform v2.0.0 - Quick Launcher           |
|        Production-Ready Wakeword Detection Training System          |
|                                                                      |
+----------------------------------------------------------------------+

System Information:
"""
    print(banner, flush=True)

    # Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"  [*] Python: {py_version}")

    # PyTorch version
    try:
        import torch
        print(f"  [*] PyTorch: {torch.__version__}")
    except ImportError:
        print(f"  [*] PyTorch: Not installed")

    # GPU status
    if cuda_available:
        print(f"  [OK] GPU: {cuda_info}")
    else:
        print(f"  [!] GPU: {cuda_info} (CPU mode)")

    # Gradio version
    try:
        import gradio
        print(f"  [*] Gradio: {gradio.__version__}")
    except ImportError:
        print(f"  [*] Gradio: Not installed")

    print("\nProduction Features:")
    features = [
        "CMVN Normalization",
        "EMA (Exponential Moving Average)",
        "Balanced Batch Sampling",
        "Learning Rate Finder",
        "Advanced Metrics (FAH, EER, pAUC)",
        "Temperature Scaling",
        "Streaming Detection",
        "ONNX Export"
    ]
    for feature in features:
        print(f"  [+] {feature}")

    print("\n" + "-" * 70)
    print("Starting application...")
    print("-" * 70 + "\n")

if __name__ == "__main__":
    # Print banner
    print_banner()

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Create necessary directories
    try:
        create_directories()
    except Exception as e:
        print(f"⚠️  Warning: Could not create directories: {e}")

    # Import and launch app
    try:
        from src.ui.app import launch_app

        print("Initializing Gradio interface...")
        print("Please wait while the application loads...\n")

        # Launch with default settings
        launch_app(
            server_name="0.0.0.0",
            server_port=None,  # Auto-find port 7860-7870
            share=False,
            inbrowser=True
        )

    except KeyboardInterrupt:
        print("\n\n✋ Application stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error launching application: {e}")
        print("\nTroubleshooting:")
        print("  1. Check that all requirements are installed: pip install -r requirements.txt")
        print("  2. Verify Python version: python --version (requires 3.8+)")
        print("  3. Check CUDA installation: nvidia-smi (for GPU support)")
        print(f"  4. See README.md for more help")
        sys.exit(1)