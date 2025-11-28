# 🚀 Industrial-Grade Upgrade: Feature Usage Guide

This guide provides step-by-step instructions for using the new "Google-Tier" features: **Quantization Aware Training (QAT)**, **Knowledge Distillation**, **Triplet Loss**, and the **Judge Server**.

These features are designed to move your wakeword model from "hobbyist" to "production-ready."

---

## 1. Quantization Aware Training (QAT)
**Best for:** Deploying to ESP32, Arduino, or other low-power microcontrollers.

QAT simulates the precision loss of 8-bit integers (INT8) during training, allowing the network to adapt. Without QAT, converting a model to INT8 often destroys accuracy.

### How to Use
1.  **Edit Configuration**:
    Open `src/config/defaults.py` (or your YAML config) and set:
    ```python
    config.qat.enabled = True
    config.qat.start_epoch = 5  # Start QAT after 5 epochs of normal training
    config.qat.backend = 'fbgemm'  # Use 'qnnpack' for ARM/Android
    ```

2.  **Train**:
    Run training as normal. The Trainer will automatically wrap your model.
    ```bash
    python run.py
    ```

3.  **Export**:
    When you export the model (Panel 5), the QAT-trained weights will be ready for INT8 conversion with minimal loss.

---

## 2. Knowledge Distillation (The Teacher)
**Best for:** Boosting the accuracy of small models (MobileNet) by mimicking a massive model (Wav2Vec 2.0).

### How to Use
1.  **Requirements**:
    Ensure you have `transformers` installed (included in `requirements.txt`).

2.  **Edit Configuration**:
    ```python
    config.distillation.enabled = True
    config.distillation.teacher_architecture = 'wav2vec2'
    config.distillation.temperature = 2.0  # Softens predictions
    config.distillation.alpha = 0.5        # Balance between Student loss and Teacher loss
    ```

3.  **Training**:
    The `DistillationTrainer` will automatically:
    - Download/Load the Wav2Vec 2.0 model (Teacher).
    - Freeze the Teacher.
    - Pass audio through both models.
    - Minimize the difference between their outputs.

    *Note: Training will use more VRAM because two models are in memory.*

---

## 3. Triplet Loss (Metric Learning)
**Best for:** Reducing false positives from phonetically similar words (e.g., "Hey Cat" vs "Hey Katya").

Instead of just classifying "Yes/No", Triplet Loss forces the model to learn a "map" where valid wakewords are clustered tightly together.

### How to Use
1.  **Edit Configuration**:
    ```python
    config.loss.loss_function = 'triplet_loss'
    config.loss.triplet_margin = 1.0
    ```

2.  **Understanding the Behavior**:
    - Training might look different. "Accuracy" might fluctuate, but the **separation** between classes is improving.
    - This is best used in a **fine-tuning phase** (Phase 2 of training) after the model has learned basic features.

---

## 4. "The Judge" Server (Stage 2 Verification)
**Best for:** A home server (Raspberry Pi 4 / Proxmox) that double-checks every wake event to prevent false alarms.

This is a standalone service that runs the heavy Wav2Vec 2.0 model.

### Setup & Installation
1.  **Navigate to Server Directory**:
    ```bash
    cd server
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Server**:
    ```bash
    uvicorn app:app --host 0.0.0.0 --port 8000
    ```

### Using with Docker
Perfect for Proxmox or Home Assistant setups.

1.  **Build Image**:
    ```bash
    docker build -f server/Dockerfile -t wakeword-judge .
    ```

2.  **Run Container**:
    ```bash
    docker run -p 8000:8000 wakeword-judge
    ```

### Testing the Endpoint
You can send a POST request with an audio file to verify a detection.

```bash
curl -X POST "http://localhost:8000/verify" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/recording.wav"
```

**Response**:
```json
{
  "prediction": 1,
  "confidence": 0.98,
  "label": "wakeword"
}
```

---

## 🎯 Recommended "Pro" Workflow

For the ultimate robust system, combine these features:

1.  **Train the Edge Model (Sentry)**:
    - Enable **Distillation** (Teacher: Wav2Vec2).
    - Enable **QAT**.
    - Train `MobileNetV3` or `TinyConv`.
    - Export to INT8.
    - *Deploy this to your ESP32 satellite devices.*

2.  **Deploy the Judge**:
    - Run the Docker container on your central server.

3.  **Runtime Logic**:
    - **ESP32** hears sound -> Runs INT8 Model.
    - If Confidence > 0.7 -> **Wake Up** (Fast!).
    - **ESP32** sends audio buffer to **Judge Server**.
    - **Judge** verifies.
    - If Judge says "Fake" -> Cancel command (High Precision).

