"""
Installation Verification Script
Checks all dependencies and system requirements
"""
import sys
import subprocess
from pathlib import Path


def print_header(text: str):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)


def print_status(check: str, passed: bool, message: str = ""):
    """Print check status"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {check}")
    if message:
        print(f"       {message}")


def check_python_version():
    """Check Python version"""
    print_header("Python Version")
    major, minor = sys.version_info[:2]
    version_str = f"{major}.{minor}"
    passed = (major == 3 and minor >= 8 and minor <= 11)

    print(f"Python Version: {sys.version}")
    print_status(
        "Python 3.8-3.11",
        passed,
        f"Found {version_str}" if passed else f"Found {version_str}, need 3.8-3.11"
    )
    return passed


def check_cuda():
    """Check CUDA availability"""
    print_header("CUDA & GPU Check")

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print_status("PyTorch installed", True)

        if cuda_available:
            device_count = torch.cuda.device_count()
            cuda_version = torch.version.cuda
            cudnn_version = torch.backends.cudnn.version()

            print(f"\nCUDA Version: {cuda_version}")
            print(f"cuDNN Version: {cudnn_version}")
            print(f"GPU Count: {device_count}")

            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}: {props.name}")
                print(f"  Compute Capability: {props.major}.{props.minor}")
                print(f"  Total Memory: {props.total_memory / (1024**3):.2f} GB")

            print_status("CUDA Available", True, f"{device_count} GPU(s) detected")

            # Check compute capability
            props = torch.cuda.get_device_properties(0)
            compute_ok = (props.major >= 6)
            print_status(
                "Compute Capability >= 6.0",
                compute_ok,
                f"Found {props.major}.{props.minor}"
            )

            return cuda_available and compute_ok
        else:
            print_status("CUDA Available", False, "No CUDA devices found")
            print("\n⚠️  GPU is MANDATORY for this platform!")
            print("Please ensure:")
            print("  1. NVIDIA GPU is installed")
            print("  2. CUDA Toolkit is installed")
            print("  3. GPU drivers are up to date")
            print("  4. PyTorch is installed with CUDA support")
            return False

    except ImportError:
        print_status("PyTorch installed", False, "torch not found")
        return False
    except Exception as e:
        print_status("CUDA Check", False, str(e))
        return False


def check_core_packages():
    """Check core package installations"""
    print_header("Core Packages")

    packages = {
        "torch": "PyTorch",
        "torchaudio": "TorchAudio",
        "torchvision": "TorchVision",
        "gradio": "Gradio",
        "librosa": "Librosa",
        "numpy": "NumPy",
        "pandas": "Pandas",
        "sklearn": "scikit-learn",
        "matplotlib": "Matplotlib",
        "onnx": "ONNX",
        "tqdm": "tqdm",
        "yaml": "PyYAML",
    }

    all_passed = True
    for module_name, display_name in packages.items():
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            print_status(display_name, True, f"v{version}")
        except ImportError:
            print_status(display_name, False, "Not installed")
            all_passed = False

    return all_passed


def check_audio_packages():
    """Check audio processing packages"""
    print_header("Audio Processing Packages")

    packages = {
        "soundfile": "SoundFile",
        "sounddevice": "SoundDevice",
        "scipy": "SciPy",
    }

    all_passed = True
    for module_name, display_name in packages.items():
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            print_status(display_name, True, f"v{version}")
        except ImportError:
            print_status(display_name, False, "Not installed")
            all_passed = False

    return all_passed


def check_optional_packages():
    """Check optional packages"""
    print_header("Optional Packages")

    packages = {
        "tensorboard": "TensorBoard",
        "pytest": "pytest",
        "colorama": "colorama",
        "psutil": "psutil",
    }

    for module_name, display_name in packages.items():
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            print_status(display_name, True, f"v{version}")
        except ImportError:
            print_status(display_name, False, "Not installed (optional)")

    return True  # Optional packages don't affect overall result


def check_directory_structure():
    """Check project directory structure"""
    print_header("Project Structure")

    required_dirs = [
        "src",
        "src/config",
        "src/data",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/export",
        "src/ui",
        "data",
        "data/raw",
        "models",
        "logs",
    ]

    all_passed = True
    project_root = Path(__file__).parent

    for dir_path in required_dirs:
        full_path = project_root / dir_path
        exists = full_path.exists()
        print_status(dir_path, exists)
        if not exists:
            all_passed = False

    return all_passed


def check_cuda_memory():
    """Check GPU memory"""
    print_header("GPU Memory Check")

    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            props = torch.cuda.get_device_properties(0)
            total_memory = props.total_memory / (1024**3)

            print(f"Total GPU Memory: {total_memory:.2f} GB")

            # Check if sufficient memory
            sufficient = total_memory >= 4.0
            print_status(
                "Sufficient GPU Memory (>=4GB)",
                sufficient,
                f"{total_memory:.2f} GB" + (" - May need small batch sizes" if total_memory < 8 else "")
            )

            # Test allocation
            try:
                test_tensor = torch.randn(1000, 1000, device=device)
                del test_tensor
                torch.cuda.empty_cache()
                print_status("GPU Memory Allocation Test", True)
                return sufficient
            except Exception as e:
                print_status("GPU Memory Allocation Test", False, str(e))
                return False
        else:
            print_status("GPU Memory Check", False, "No CUDA device")
            return False
    except Exception as e:
        print_status("GPU Memory Check", False, str(e))
        return False


def main():
    """Run all verification checks"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║       Wakeword Training Platform - Installation Check       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    results = {
        "Python Version": check_python_version(),
        "CUDA & GPU": check_cuda(),
        "Core Packages": check_core_packages(),
        "Audio Packages": check_audio_packages(),
        "Optional Packages": check_optional_packages(),
        "Project Structure": check_directory_structure(),
        "GPU Memory": check_cuda_memory(),
    }

    # Summary
    print_header("Verification Summary")

    passed_count = sum(results.values())
    total_count = len(results)

    for check, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")

    print(f"\n{passed_count}/{total_count} checks passed")

    if all(results.values()):
        print("\n" + "="*60)
        print("🎉 ALL CHECKS PASSED!")
        print("="*60)
        print("\nYour system is ready for wakeword training.")
        print("\nTo start the platform:")
        print("  python src/ui/app.py")
        print("\nor:")
        print("  python -m src.ui.app")
        return 0
    else:
        print("\n" + "="*60)
        print("⚠️  SOME CHECKS FAILED")
        print("="*60)

        if not results["CUDA & GPU"]:
            print("\n❌ CRITICAL: GPU/CUDA is MANDATORY for this platform")
            print("Please install CUDA and ensure GPU is available")

        if not results["Core Packages"]:
            print("\n❌ Install missing packages:")
            print("  pip install -r requirements.txt")

        if not results["Project Structure"]:
            print("\n❌ Project structure incomplete")
            print("Please ensure all directories exist")

        print("\nRe-run this script after fixing issues:")
        print("  python verify_installation.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())