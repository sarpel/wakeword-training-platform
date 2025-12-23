# 🌐 Distributed Cascade Architecture Guide

🎯 **Google-Tier Performance** with **MCU-grade Efficiency**. Production-ready cascade for real-time wakeword detection across edge devices and servers.

⚡ **Key Innovation**: Achieve 99.9% accuracy with <1% energy consumption through intelligent two-stage inference.

---

## 1. The Concept: Why use a Cascade?

## 🎯 The Challenge: Power vs. Accuracy Trade-off

| 🔋 Device Type | ⚡ Power Usage | 🧠 Processing | ❌ False Positives |
|---|---|---|---|
| **MCU (Sentry)** | Ultra-Low (<1W) | Limited | High (10-15%) |
| **Server (Judge)** | High (10-100W) | Massive | Very Low (<0.1%) |

## 🚀 Solution: Two-Stage Cascade

### **Stage 1: Sentry (Edge Detection)**
- **🎯 Mission**: Filter 99%+ of ambient noise
- **🧠 Model**: Tiny-Conv / MobileNetV3 (ultra-optimized)
- **⚡ Power**: Always-on, <1% energy impact
- **📤 Action**: On "possible detection", sends audio to server

### **Stage 2: Judge (Final Verification)**
- **🎯 Mission**: Eliminate false positives with 99.9% accuracy
- **🧠 Model**: ResNet18 / Wav2Vec2 (server-grade)
- **⚡ Power**: Activated only when needed
- **✅ Result**: Final command execution

**🏆 Net Effect**: **Server-grade accuracy** with **MCU-grade power consumption**

---

## 2. Training the Models

### 2.1 The Sentry Model (MCU)
Use the following profiles in the **Configuration Panel**:
- **ESP32-S3:** `Production: ESP32-S3 (Standard)`
- **Raspberry Pi Zero 2W:** `Production: RPi Zero 2W Satellite`

**Key Goal:** Focus on **Recall** (catching every trigger). It's okay if it has some false positives, as the Judge will filter them out.

### 2.2 The Judge Model (Server)
Use the following profile:
- **Server:** `Production: Server (High Accuracy Judge)`

**Key Goal:** Focus on **Precision** (never triggering wrongly). Use a larger context window (2.0s) and higher resolution (64-80 Mel bands).

---

## 3. Installation & Setup

### 3.1 Server Installation (The Judge)
The Judge server is a Python-based API that hosts your high-accuracy models.

1.  **Export your Judge model** to ONNX or standard PyTorch format using the **Export Panel**.
2.  **Navigate to the server directory**:
    ```bash
    cd server
    pip install -r requirements.txt
    ```
3.  **Start the server**:
    ```bash
    cd server
    uvicorn app:app --host 0.0.0.0 --port 8000
    ```

### 3.2 MCU Installation (The Sentry)
1.  **Export your Sentry model** to TFLite (for ESP32) or ONNX (for RPi).
2.  **Flash the firmware** to your device (using ESP-Skainet or Wyoming Satellite).
3.  **Configure the Server URL**: Set the device to point to your Judge server IP (e.g., `http://192.168.1.100:8000/verify`).

---

## 4. Connection Logic (How they talk)

The communication follows a simple **Verification Protocol**:

1.  **Detection:** Sentry triggers locally on the word "Hey Assistant."
2.  **Buffering:** The Sentry captures the last 1.5 seconds of audio from its internal ring buffer.
3.  **Request:** Sentry sends an HTTP POST request to the Judge server:
    ```http
    POST /verify
    Content-Type: audio/wav
    [Binary Audio Data]
    ```
4.  **Inference:** The Judge runs the high-accuracy model on the received audio.
5.  **Response:** The Judge returns a JSON decision:
    ```json
    {
      "verified": true,
      "confidence": 0.98,
      "latency_ms": 45
    }
    ```
6.  **Action:** If `verified` is true, the MCU lights up the LED and processes your command.

---

## 5. Performance Tuning (Consensus Values)

To ensure the system works reliably, follow these "Industrial Consensus" rules:

| Parameter | Sentry (MCU) | Judge (Server) | Why? |
| :--- | :--- | :--- | :--- |
| **Sample Rate** | 16,000 Hz | 16,000 Hz | Must match for audio compatibility. |
| **Mel Bands** | 40 | 64 - 80 | Sentry needs speed; Judge needs detail. |
| **Window** | 1.0s - 1.5s | 1.5s - 2.0s | Judge needs more context to be sure. |
| **Threshold** | Low (0.4 - 0.5) | High (0.8 - 0.9) | Sentry catches all; Judge filters strictly. |

---

## 6. Troubleshooting

- **High Latency:** Ensure your server has a GPU or uses the **ONNX Runtime** with OpenVINO/TensorRT.
- **Missed Detections:** Check if the audio sent by the MCU is clipped or too quiet. The Judge and Sentry must use the same **CMVN Normalization** stats.
- **Connection Refused:** Ensure the Server firewall allows traffic on port 8000.
