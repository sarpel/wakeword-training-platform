# Wakeword Training Platform

**Production-Ready Wakeword Detection Training with GPU Acceleration and Gradio UI**

A complete, user-friendly platform for training custom wakeword detection models with state-of-the-art features and a beautiful web interface. Now features a "Google-Tier" Distributed Cascade Architecture.

---

## 🏗️ "Google-Tier" Architecture (New!)

The system now supports a 3-stage distributed architecture for enterprise-grade performance:

1.  **Stage 1: The Sentry (Edge)**
    *   **Role:** Ultra-low power wake-up.
    *   **Goal:** High Recall (Never miss a command).
    *   **Tech:** Quantized MobileNetV3 + QAT (Quantization Aware Training).
    *   **Constraint:** < 200KB RAM.

2.  **Stage 2: The Judge (Local Server)**
    *   **Role:** False Positive Rejection.
    *   **Goal:** High Precision (Filter out "Hey Cat" vs "Hey Katya").
    *   **Tech:** **Wav2Vec 2.0** Transformer running on a local server/container.
    *   **Input:** Verifies the 1.5s audio buffer from Stage 1.

3.  **Stage 3: The Teacher (Training)**
    *   **Role:** Knowledge Distillation.
    *   **Tech:** A massive Wav2Vec 2.0 model "teaches" the smaller mobile models to behave like it.

---

## Features at a Glance

### 🎨 Beautiful Web Interface
- Six intuitive panels guide you through the complete workflow
- Real-time training visualization with live metrics
- Interactive evaluation with microphone testing
- No command-line required - everything in your browser!

### 🚀 Production-Ready Training
- **Knowledge Distillation**: Train small models to match large transformer accuracy.
- **Quantization Aware Training (QAT)**: Prepare models for INT8 edge deployment without losing accuracy.
- **Triplet Loss**: Metric learning for better separation of similar sounding words.
- **CMVN & EMA**: Stabilization techniques for robust real-world performance.

### 📊 Advanced Metrics
- **FAH (False Alarms per Hour)**: Production metric that matters
- **EER (Equal Error Rate)**: Research-standard metric
- **Operating Point Selection**: Find the best threshold for your use case

### 🛡️ The Judge Server
- Standalone Docker container for false positive rejection.
- FastAPI-based inference engine.
- Plugs into Home Assistant or other smart home stacks.

---

## Quick Start

### Prerequisites

- **Python**: 3.10 or higher
- **CUDA**: 11.8 or higher (for GPU acceleration)
- **GPU**: NVIDIA GPU with 6GB+ VRAM recommended
- **RAM**: 16GB+ recommended

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/sarpel/wakeword-training-platform.git
   cd wakeword-training-platform
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: For PyTorch with CUDA 11.8, use:
   ```bash
   pip install torch==2.1.2+cu118 torchaudio==2.1.2+cu118 torchvision==0.16.2+cu118 --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Launch the Application**
   ```bash
   python run.py
   ```

   The application will open in your web browser at `http://localhost:7860`

---

## The Judge Server (Stage 2)

To run the False Positive Rejection server:

1.  **Install Server Dependencies**:
    ```bash
    pip install -r server/requirements.txt
    ```

2.  **Run Server**:
    ```bash
    uvicorn server.app:app --host 0.0.0.0 --port 8000
    ```

3.  **Docker Deployment**:
    ```bash
    docker build -f server/Dockerfile -t wakeword-judge .
    docker run -p 8000:8000 wakeword-judge
    ```

---

## How to Use

### Panel 1: Dataset Management
**Scan & Organize Your Audio Files**. Split into positive/negative samples.

### Panel 2: Configuration
**Choose Your Model & Settings**.
- Enable **Distillation** to use a Teacher model.
- Enable **QAT** if deploying to microcontrollers.

### Panel 3: Training
**Train Your Model with One Click**.
- Monitor F1 Score and Loss.
- Watch for "New best model" notifications.

### Panel 4: Evaluation
**Test Your Model**.
- Use **File Evaluation** or **Live Microphone**.
- Check **FAH** (False Alarms per Hour).

### Panel 5: Model Export
**Deploy Your Model**.
- Export to **ONNX** for general use.
- Export to **Quantized ONNX** for edge devices.

---

## Project Structure

```
wakeword-training-platform/
├── README.md                 # This file
├── TECHNICAL_FEATURES.md     # Detailed technical documentation
├── run.py                    # Quick launcher
├── requirements.txt          # Core dependencies
├── server/                   # "The Judge" Server
│   ├── app.py                # FastAPI app
│   ├── inference_engine.py   # Wav2Vec2 Inference
│   └── Dockerfile            # Container config
├── src/                      # Source code
│   ├── models/               # Architectures (ResNet, MobileNet, Wav2Vec)
│   ├── training/             # Trainer, Distillation, QAT
│   └── ...
└── ...
```

---

## License

MIT License - See LICENSE file for details

---

**Happy Training! 🚀**