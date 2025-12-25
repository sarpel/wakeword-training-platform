# ⚙️ Config Presets & Training Profiles Guide

🎯 **Production-Optimized Configuration Templates** for different deployment targets. Each preset is battle-tested for both **Standard Training** and **Hyperparameter Optimization (HPO)**.

💡 **Quick Tip**: All presets auto-configure: learning rates, batch sizes, and model architecture for optimal performance.

## ⚡ Hardware Selection Matrix

| 🎯 Target | ⚙️ Recommended Profile | 💾 Memory | 🧠 Architecture | 📊 Performance |
|---|---|---|---|---|
| **ESP32-S3-BOX-3** | `Production: ESP32-S3 (PSRAM)` | **8-16MB PSRAM** | MobileNetV3 | 95%+ Accuracy |
| **ESP32-S3 (PSRAM)** | `Production: ESP32-S3 (PSRAM)` | **PSRAM Required** | MobileNetV3 | High Accuracy |
| **M5Stack Atom Echo** | `Production: MCU (No-PSRAM)` | **Internal RAM only**| TinyConv | Ultra-Fast |
| **ESP32 / ESP32-C3** | `Production: MCU (No-PSRAM)` | **<500KB SRAM** | TinyConv | Real-time |
| **RPi Zero 2W/3/4**| `Production: RPi Satellite` | Linux System | MobileNetV3 | Server-Grade |
| **Desktop/GPU** | `Production: Ultimate Accuracy` | High VRAM | ResNet18 | Best-in-Class |

🔥 **Performance**: All models optimized for <10% FNR on target hardware.

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

### 4. Production: RPi Zero 2W Satellite
**Best for:** Raspberry Pi Zero 2W, 3, or 4 running as voice satellites.
- **Architecture:** `MobileNetV3-Small`
- **Why this?** Optimized for the Cortex-A53 processor and Wyoming protocol.

### 5. Production: x86_64 (Ultimate Accuracy)
**Best for:** Running detection on powerful hardware (PC, Home Assistant Blue/Yellow, NUC).
- **Architecture:** `ResNet18`
- **Why this?** Uses 80 Mel bands and high-resolution features (2.0s context). It is too "heavy" for microcontrollers but provides near-perfect detection for server-side or desktop use cases.

---

## 🛠️ Utility Profiles

### 1. Utility: Small Dataset (<10k)
Optimized for training when you have very limited data. Uses high dropout (0.5) and aggressive augmentation to prevent overfitting.

### 2. Utility: Fast Training (Prototyping)
Used for quick iterations. Uses a high learning rate and fewer epochs.

---

## 🛠️ Usage Note: Training vs. HPO
- **Training:** Select a preset as your starting point. It pre-fills optimal values for learning rate, batch size, and architecture.
- **HPO (Auto-Tuning):** Use these profiles as the "Baseline". The HPO engine will take these values and attempt to find even better ones by searching around these industry-standard defaults.
