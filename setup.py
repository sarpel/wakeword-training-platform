"""
Wakeword Training Platform Setup
Production-Ready Wakeword Detection Training System
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "Production-ready wakeword detection training platform with GPU acceleration and Gradio UI"

setup(
    name="wakeword-training-platform",
    version="2.0.0",
    author="Wakeword Platform Team",
    author_email="contact@example.com",
    description="Production-ready GPU-accelerated wakeword training platform with advanced features and Gradio UI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sarpel/wakeword-training-platform",
    project_urls={
        "Bug Reports": "https://github.com/sarpel/wakeword-training-platform/issues",
        "Documentation": "https://github.com/sarpel/wakeword-training-platform/blob/main/TECHNICAL_FEATURES.md",
        "Source": "https://github.com/sarpel/wakeword-training-platform",
    },
    packages=find_packages(exclude=["tests", "docs", "examples"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Environment :: GPU :: NVIDIA CUDA :: 11.8",
        "Framework :: Gradio",
    ],
    keywords=[
        "wakeword",
        "wake-word",
        "hotword",
        "voice-activation",
        "speech-recognition",
        "keyword-spotting",
        "pytorch",
        "deep-learning",
        "audio-classification",
        "gradio",
        "gpu-acceleration",
        "production-ready",
    ],
    python_requires=">=3.9",
    install_requires=[
        # Core Deep Learning
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "torchvision>=0.16.0",
        # UI Framework
        "gradio>=4.20.0",
        "websockets>=10.0",
        # Audio Processing
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "sounddevice>=0.4.0",
        "resampy>=0.4.0",
        "audioread>=3.0.0",
        # Data Processing & ML
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        # Visualization
        "matplotlib>=3.7.0",
        "plotly>=5.17.0",
        "seaborn>=0.12.0",
        # Model Export
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0",
        # Utilities
        "tqdm>=4.65.0",
        "pyyaml>=6.0.0",
        "tensorboard>=2.13.0",
        "scipy>=1.10.0",
        "psutil>=5.9.0",
        "colorama>=0.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
        ],
        "gpu": [
            "onnxruntime-gpu>=1.15.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "wakeword-train=src.ui.app:launch_app",
        ],
    },
    include_package_data=True,
    package_data={
        "src": [
            "config/*.yaml",
            "ui/assets/*",
        ],
    },
    zip_safe=False,
    platforms=["any"],
)

# Post-installation message
print(
    """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   Wakeword Training Platform v2.0.0 Installed Successfully! ║
║                                                              ║
║   Production Features:                                       ║
║   ✓ CMVN Normalization                                       ║
║   ✓ EMA (Exponential Moving Average)                         ║
║   ✓ Balanced Batch Sampling                                  ║
║   ✓ Learning Rate Finder                                     ║
║   ✓ Advanced Metrics (FAH, EER, pAUC)                        ║
║   ✓ Temperature Scaling                                      ║
║   ✓ Streaming Detection                                      ║
║   ✓ ONNX Export                                              ║
║                                                              ║
║   Quick Start:                                               ║
║   1. Prepare your audio data in data/raw/positive and        ║
║      data/raw/negative folders                               ║
║   2. Run: python run.py                                      ║
║   3. Open your browser at http://localhost:7860              ║
║                                                              ║
║   Documentation:                                             ║
║   - README.md: Quick start guide                             ║
║   - TECHNICAL_FEATURES.md: Comprehensive technical docs      ║
║                                                              ║
║   For help: https://github.com/sarpel/wakeword-training-platform║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
)
