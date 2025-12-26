# Wakeword Training Platform [![PR Status](https://img.shields.io/github/pull-request/status/sarpel/wakeword-training-platform/13)](https://github.com/sarpel/wakeword-training-platform/pull/13) [![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Production-ready platform for training custom wakeword detection models with **GPU acceleration**, **advanced optimizations**, and a modern web interface. Features enterprise-grade **Distributed Cascade Architecture** for real-time deployment.

🚀 **Current Version**: v4.0 - Production Release
🔧 **GPU Support**: CUDA 11.8+ with Mixed Precision
🌐 **Deployment**: ONNX, TensorFlow Lite, Raspberry Pi

---

## 📚 Quick Navigation

| 📖 Documentation | 🔧 Configuration | 🎯 Usage |
|---|---|---|
| **[📘 Complete Guide](DOCUMENTATION.md)** | **[⚙️ Presets](CONFIG_PRESETS_GUIDE.md)** | **[🚀 Quick Start](#--quick-start)** |
| User Guide & Reference | GPU/RPi Optimization | Training & Deployment |

**🔍 Need help?** Check our [Technical Features Guide](TECHNICAL_FEATURES.md) for CMVN, EMA, and FAH metrics.

---

## 🚀 Quick Start

### Prerequisites
- **Python**: 3.10+
- **CUDA**: 11.8+ (for GPU acceleration)
- **GPU**: NVIDIA GPU with 6GB+ VRAM recommended

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/sarpel/wakeword-training-platform.git
    cd wakeword-training-platform
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For PyTorch with CUDA 11.8, see [DOCUMENTATION.md](DOCUMENTATION.md).*

3.  **Launch the Application**
    ```bash
    python run.py
    ```
    The application will open at `http://localhost:7860`

### 🚀 Quick Start (Docker - Recommended)

For a consistent environment across Windows and Linux:

1.  **Configure Environment**
    ```bash
    cp .env.example .env
    # Edit .env to set your QUANTIZATION_BACKEND (fbgemm for Win, qnnpack for Linux)
    ```

2.  **Launch via Docker Compose**
    ```bash
    docker-compose up -d
    ```

3.  **Access Services**
    - **Dashboard**: `http://localhost:7860`
    - **Inference Server**: `http://localhost:8000`
    - **Jupyter Lab**: `http://localhost:8888`
    - **TensorBoard**: `http://localhost:6006`

---

## 📂 Data Preparation

The platform expects audio files in the following structure:
- `data/raw/positive/`: Put your wakeword audio files here (.wav, .flac, .mp3).
- `data/raw/negative/`: Put background noise and non-wakeword speech here.

The system will automatically create these directories on first run.

---

## 🏗️ Distributed Cascade Architecture

**Production-Ready 3-Stage Pipeline** for real-time wakeword detection:

| ⚡ Stage | 🎯 Purpose | 🧠 Model | 📊 Metrics |
|---|---|---|---|
| **Sentry (Edge)** | Always-On Detection | MobileNetV3 + QAT | <1% FNR, <0.1% Energy |
| **Judge (Local)** | False Positive Filtering | Wav2Vec 2.0 | >99% Accuracy |
| **Teacher (Cloud)** | Knowledge Distillation | Teacher-Student | 10x Faster Training |

**🔬 Advanced Features**: CMVN, EMA, Mixed Precision, FAH Metrics
📖 **[Architecture Deep Dive](DOCUMENTATION.md#distributed-cascade-architecture)**

---

##  What's New in v4.0

- **📉 New**: Focal Loss implementation for superior hard-negative handling
- **⚡ New**: QAT Accuracy Recovery pipeline (FP32 baseline to INT8 fine-tuning)
- **📏 New**: Model Size Insight & Platform Constraints validation for Edge deployment
- **✨ New**: Advanced GPU acceleration with Mixed Precision training
- **🚀 New**: Comprehensive HPO (Hyperparameter Optimization) system
- **📦 New**: Production-ready ONNX and TFLite export
- **🎯 New**: Knowledge Distillation for 10x faster edge deployment
- **🔧 New**: Raspberry Pi optimized models and configs

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

---

**🚀 Happy Training!** ⭐ **Star us on GitHub!**
