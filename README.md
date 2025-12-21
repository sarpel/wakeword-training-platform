# Wakeword Training Platform

A complete, user-friendly platform for training custom wakeword detection models with state-of-the-art features and a beautiful web interface. Now features a "Google-Tier" Distributed Cascade Architecture.

---

## 📚 Documentation

Detailed documentation has been consolidated into [DOCUMENTATION.md](DOCUMENTATION.md).

- **[User Guide](DOCUMENTATION.md#part-1-user-guide)**: Configuration, Training, and Feature Usage.
- **[Technical Reference](DOCUMENTATION.md#part-2-technical-reference)**: Architecture, Algorithms, and Developer Guidelines.

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

---

## 🏗️ Architecture Overview

The system supports a 3-stage distributed architecture:
1.  **The Sentry (Edge)**: Ultra-low power (MobileNetV3 + QAT).
2.  **The Judge (Local Server)**: False Positive Rejection (Wav2Vec 2.0).
3.  **The Teacher (Training)**: Knowledge Distillation.

See [Technical Reference](DOCUMENTATION.md#part-2-technical-reference) for details.

---

## 📄 License

MIT License - See LICENSE file for details

**Happy Training! 🚀**