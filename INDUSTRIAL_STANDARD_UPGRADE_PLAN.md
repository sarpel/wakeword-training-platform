# Industrial-Grade Wakeword Project Upgrade Plan: The "Google-Tier" Roadmap

This document outlines the architectural vision for the "Hey Katya" wakeword system. It moves beyond a simple standalone model to a **Distributed Cascade Architecture** that leverages cloud-grade training (Colab A100), local server verification (Proxmox), and edge efficiency (ESP32).

---

## 🛡️ Core Design Directives (MANDATORY)

**To ensure stability and backward compatibility, all implementing agents must adhere to these rules:**

1.  **Additive Implementation:** New features (QAT, Distillation, Transformers) must be implemented as **extensions**, not replacements. The existing `ResNet18` + `CrossEntropy` pipeline must remain fully functional.
2.  **Opt-In Configuration:** All new features must be controlled via flags in the configuration (e.g., `use_qat: bool`, `distillation_enabled: bool`). If these flags are missing or False, the system behaves exactly as it did before.
3.  **Modular Separation:**
    *   Do not clutter `src/training/trainer.py` with massive if/else blocks. Create subclasses (e.g., `DistillationTrainer(Trainer)`) or mixins.
    *   New massive models (Wav2Vec 2.0) should reside in their own module (e.g., `src/models/transformers.py`) to avoid heavy dependency requirements for basic edge users.
4.  **Interface Consistency:** New models and components must adhere to the existing interfaces (`forward(x)`, `train_step()`) so they plug into the existing GUI and evaluation pipelines without refactoring.

---

## 🏗️ Architecture Overview: The "Home Lab Cascade"

The system is divided into three distinct stages, optimizing for the strengths of each hardware tier.

### Stage 1: The Sentry (ESP32 / Edge)
*   **Role:** Always-On Wake-up.
*   **Goal:** **High Recall** (Never miss a command). It is acceptable to trigger accidentally on "Hey Cat" occasionally.
*   **Model:** **MobileNetV3 (INT8 Quantized)** + **Voice Activity Detection (VAD)**.
*   **Constraint:** < 200KB RAM, Real-time (<50ms latency).

### Stage 2: The Judge (Proxmox Server / Desktop GPU)
*   **Role:** False Positive Rejection.
*   **Goal:** **High Precision** (Never allow a false trigger).
*   **Model:** **Wav2Vec 2.0 (Fine-Tuned)**.
*   **Input:** Receives the 1-2 second audio buffer from Stage 1 upon trigger.
*   **Constraint:** Latency < 300ms (LAN speed).

### Stage 3: The Teacher (Google Colab A100)
*   **Role:** The Brain / Training Ground.
*   **Goal:** Train the "God Model" and distill its knowledge down to the Sentry.
*   **Asset:** A100 GPU (Colab Pro+).

---

## 🚀 Phase 1: The Sentry (Edge Optimization)

### 1. Quantization Aware Training (QAT)
**What is it?**  
Simulates 8-bit integer (INT8) precision during training.
**Implementation Strategy (Modular):**
- **Do NOT** modify `src/models/architectures.py` directly.
- **Create** `src/models/quantized_architectures.py` or a wrapper function `prepare_model_for_qat(model)`.
- **Config:** Add `qat_enabled` to `TrainingConfig`.
- **Workflow:** If `qat_enabled` is True, the Trainer wraps the model dynamically. The original training flow remains untouched for non-QAT runs.

### 2. Voice Activity Detection (VAD)
**What is it?**  
A tiny, non-AI energy detector (or a micro-model like Silero VAD) that runs *before* the wakeword model.
**Implementation Strategy:**
- **Inference Only:** This logic primarily lives in the deployment code (C++ firmware or Python inference script), not the training loop.
- **Data Prep:** Optionally add a `VADFilter` class in `src/data/preprocessing.py` to clean datasets, but keep it optional.

### 3. Streaming Inference Simulation
**What is it?**  
Training the model on "sliding windows" of audio rather than perfect 1.0s clips.
**Implementation Strategy:**
- **Augmentation:** Add `RandomTimeShift` to `src/data/augmentation.py`.
- **Backward Compatibility:** This is just a new augmentation type. Existing datasets work fine.

---

## 🧠 Phase 2: The Teacher (Colab A100 & Distillation)

### 4. The "God Model": Fine-Tuning Wav2Vec 2.0
**What is it?**  
Using Facebook's Wav2Vec 2.0 (XLSR-53) as the ultimate feature extractor.
**Implementation Strategy:**
- **New Module:** `src/models/huggingface.py`.
- **Dependency:** `transformers` library (make import optional to not break lightweight envs).
- **Integration:** Create a `Wav2VecWakeword` class that inherits from `nn.Module` and matches the interface of existing models.

### 5. Knowledge Distillation (Teacher -> Student)
**What is it?**  
Training the MobileNetV3 (Student) to match the Wav2Vec 2.0 (Teacher) output probabilities.
**Implementation Strategy:**
- **New Trainer:** Create `src/training/distillation_trainer.py`.
- **Inheritance:** `class DistillationTrainer(Trainer): ...`
- **Logic:** Override `compute_loss`. Calculate `StudentLoss + (alpha * DistillationLoss)`.
- **Config:** Add `DistillationConfig` to `defaults.py`.
- **Benefit:** If Distillation is disabled, the standard `Trainer` is used, ensuring 100% backward compatibility.

---

## 🛡️ Phase 3: The Judge (The Stack)

### 6. The "Stack" Implementation (Proxmox Service)
**What is it?**  
A Docker container running on your always-on server.
**Implementation Strategy:**
- **New Directory:** `server/` (Completely separate from `src/`).
- **Components:**
    - `server/app.py`: FastAPI application.
    - `server/inference_engine.py`: Loads the Wav2Vec model.
    - `server/Dockerfile`: For easy Proxmox deployment.

### 7. Metric Learning (Triplet Loss)
**What is it?**  
Training the model to cluster audio samples in geometric space.
**Implementation Strategy:**
- **Loss Factory:** Update `src/models/losses.py` to support `TripletMarginLoss`.
- **Config:** Add "triplet_loss" as an option in `LossConfig.loss_function`.
- **Sampler:** Ensure `BalancedBatchSampler` supports triplet mining (it already provides balanced classes, which is 90% of the work).

---

## 📅 Execution Roadmap for Agents

**Step 1: The Foundation (Local)**
- [ ] **Config Update:** Add `QATConfig` and `DistillationConfig` dataclasses to `defaults.py` (Optional fields).
- [ ] **QAT:** Implement `src/training/qat_utils.py` to handle model preparation without touching core architecture files.
- [ ] **Losses:** Add `TripletLoss` to `src/models/losses.py`.

**Step 2: The Teacher (Colab)**
- [ ] **New Model:** Implement `src/models/huggingface.py` (Wav2Vec2 wrapper).
- [ ] **New Trainer:** Implement `src/training/distillation_trainer.py`.

**Step 3: The Stack (Proxmox/Local)**
- [ ] **Server:** Create `server/` directory and basic FastAPI scaffold.

**Step 4: The Firmware (ESP32)**
- [ ] (Future Task) Implement the HTTP/UDP client to send audio buffers to "The Judge" upon trigger.

This plan guarantees that while we aim for "Google-Tier" performance, we maintain "Engineering-Tier" code quality: modular, safe, and robust.
