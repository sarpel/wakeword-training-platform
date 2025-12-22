# Config Presets & Training Profiles Guide

This guide details the **Industrial Standardized Training Presets** available in the configuration panel. These profiles serve as optimized baselines for both **Standard Training** and **Hyperparameter Optimization (HPO)**.

## 🚀 Quick Selection Guide (Hardware Based)

| Hardware | Recommended Profile | Memory Type | Architecture |
| :--- | :--- | :--- | :--- |
| **ESP32-S3-BOX-3** | `Production: ESP32-S3 (PSRAM)` | **PSRAM Required** | MobileNetV3 |
| **ESP32-S3 (w/ PSRAM)** | `Production: ESP32-S3 (PSRAM)` | **PSRAM Required** | MobileNetV3 |
| **M5Stack Atom Echo** | `Production: MCU (No-PSRAM)` | **Internal RAM only**| TinyConv |
| **ESP32 / ESP32-C3** | `Production: MCU (No-PSRAM)` | **Internal RAM only**| TinyConv |
| **RPi Zero 2W / 3 / 4**| `Production: RPi Zero 2W Satellite`| Linux / Wyoming | MobileNetV3 |
| **Desktop / Server** | `Production: x86_64 (Ultimate)` | High Performance | ResNet18 |

---

## 📋 Detailed Profile Descriptions

### 1. Production: ESP32-S3 (PSRAM)
**Best for:** Modern ESP32-S3 devices with external PSRAM (8MB/16MB).
- **Architecture:** `MobileNetV3-Small`
- **Why this?** It provides the highest accuracy for embedded devices. It uses a deeper neural network that requires the extra memory buffer provided by PSRAM to handle the model weights and activations during inference.
- **Quantization:** Ready for Int8 Quantization-Aware Training (QAT).

### 2. Production: MCU (No-PSRAM)
**Best for:** Older ESP32 models or small boards like **M5Stack Atom Echo** that rely strictly on internal SRAM.
- **Architecture:** `TinyConv` (DS-CNN style)
- **Why this?** It is optimized for extremely low memory footprint (<100KB Peak RAM). While slightly less robust than MobileNetV3, it is much faster and fits into the limited memory of non-PSRAM chips.

### 3. Production: Home Assistant / Wyoming
**Best for:** Compatibility with the official Home Assistant Wyoming protocol.
- **Architecture:** `MobileNetV3-Small`
- **Context:** Uses a 1.5s audio window to match standard Home Assistant voice satellite behaviors.

### 4. Production: x86_64 (Ultimate Accuracy)
**Best for:** Running detection on powerful hardware (PC, Home Assistant Blue/Yellow, NUC).
- **Architecture:** `ResNet18`
- **Why this?** Uses 80 Mel bands and high-resolution features. It is too "heavy" for microcontrollers but provides near-perfect detection for server-side or desktop use cases.

---

## 🛠️ Usage Note: Training vs. HPO
- **Training:** Select a preset as your starting point. It pre-fills optimal values for learning rate, batch size, and architecture.
- **HPO (Auto-Tuning):** Use these profiles as the "Baseline". The HPO engine will take these values and attempt to find even better ones by searching around these industry-standard defaults.
