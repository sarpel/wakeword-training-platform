"""
Panel 6: Documentation & Knowledge Base
- Comprehensive wakeword training guide
- Best practices and industry standards
- Troubleshooting guide
"""

import gradio as gr


def create_docs_panel() -> gr.Blocks:
    """
    Create Panel 6: Documentation & Knowledge Base

    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks() as panel:
        gr.Markdown("# 📚 Documentation & Knowledge Base")
        gr.Markdown(
            "Welcome to the **Sarpel Wakeword Training Platform**. "
            "This guide explains **everything** you need to know, even if you have zero experience with AI."
        )

        with gr.Tabs():
            # ==========================================================================================================
            # 1. DATASET HANDLING
            # ==========================================================================================================
            with gr.TabItem("🗃️ Dataset Handling"):
                gr.Markdown(
                    """
# 🗃️ Dataset Handling (Panel 1)

Before the AI can learn, it needs examples. This panel is where you prepare your "textbooks" for the AI.

## 📂 The Folder Structure
Your data folder is like a filing cabinet. The AI expects files in specific drawers (folders):

- **`positive/`**: Examples of the specific word you want to detect (e.g., "Hey Computer").
  - *Quantity*: 1,000+ files. The more, the better.
  - *Variety*: Men, women, children, fast, slow, accents.
- **`negative/`**: EVERYTHING ELSE. Speech that is *not* your wakeword.
  - *Quantity*: 10,000+ files.
  - *Content*: Conversations, TV audio, random words.
- **`hard_negative/`**: Tricky words that sound *almost* like your wakeword.
  - *Example*: If your word is "Hey Siri", put "Hey Serious" or "Hey City" here.
  - *Purpose*: Teaches the AI not to get confused by similar sounds.
- **`background/`**: Noise with no speech.
  - *Content*: Rain, traffic, fan noise, white noise.
  - *Purpose*: Teaches the AI to ignore the environment.
- **`rirs/` (Room Impulse Responses)**: "Acoustic fingerprints" of rooms.
  - *Explanation*: A clap sounds different in a bathroom vs. a cathedral. RIRs capture this echo.
  - *Usage*: The system mixes these with your voice to simulate different rooms.

## 🛠️ The Workflow (Step-by-Step)

### 1. 🔍 Scan Datasets
- **What it does**: Looks through your folders, counts the files, and checks if they are valid audio.
- **Result**: Generates a "Manifest" (a list of all valid files).
- **Settings**:
  - `Fast Scan`: Checked by default. Uncheck it if you want to verify every single file (slow but safe).

### 2. ⚡ Extract Features (NPY)
- **What it does**: Converts audio (sound waves) into "features" (math representations) *before* training.
- **Why do this?**:
  - **Speed**: Training becomes **40-60% faster**.
  - **Efficiency**: The CPU doesn't have to process audio every single loop.
- **How**: Click "Extract All Features to NPY". It saves `.npy` files in `data/npy`.
- **Note**: If you change your dataset, you must run this again!

### 3. ✂️ Split Datasets
- **What it does**: Divides your data into three buckets:
  1.  **Train (70%)**: The AI studies these to learn.
  2.  **Validation (15%)**: The AI takes practice exams on these to see how it's doing.
  3.  **Test (15%)**: The "Final Exam". The AI *never* sees these until the very end.
- **Why?**: To make sure the AI isn't just memorizing definitions but actually understanding concepts.
- **Action**: Click "Split Datasets" after scanning.

---
**⚠️ Note:** We removed the "Process Pipeline" button. The new rigorous standard is: **Scan → Extract → Split**. This ensures maximum reliability.
                    """
                )

            # ==========================================================================================================
            # 2. CONFIGURATION VARIABLES
            # ==========================================================================================================
            with gr.TabItem("⚙️ Configuration Logic"):
                gr.Markdown(
                    """
# ⚙️ Configuration Variables (Panel 2)

This is the "Control Room". Here is what every knob and dial actually does.

## 🎵 Audio Settings (The "Ears")

| Variable | Recommended | Description (ELI5) |
| :--- | :--- | :--- |
| **Sample Rate** | `16000` | **The resolution of sound.** Standard is 16kHz. <br>• *too low (8k)*: Sounds like a walkie-talkie. <br>• *too high (44k)*: CD quality, but wastes space/processing power for voice. |
| **Audio Duration** | `1.5` | **Attention Span.** How many seconds the AI listens to at once. <br>• Short words ("Alexa") need ~1.0s. <br>• Long phrases ("Hey Google") need 1.5s - 2.0s. |
| **Feature Type** | `mel` | **The Format.** <br>• **`mel`**: Picture of sound (Spectrogram). Best for neural networks. <br>• **`mfcc`**: compressed shape of sound. Older technique, usually worse for deep learning. |
| **n_mels** | `64` | **Vertical Resolution.** How many frequency bands we split sound into. <br>• *Higher (80+)*: More detail, more compute. <br>• *Lower (32)*: Blurry sound, faster. |
| **n_fft & hop** | `512/160` | **Frame Rate.** Controls how "smooth" the audio picture is over time. Default is optimal for 16kHz. |

## 🧠 Training Parameters (The "Brain")

| Variable | Value | Meaning & Impact |
| :--- | :--- | :--- |
| **Batch Size** | `32` / `64` | **"Flashcards per Quiz".** <br>• **High (64+)**: Fast training, stable, uses HUGE Video Memory (VRAM). <br>• **Low (8-16)**: Slower, erratic learning, but fits on smaller GPUs. |
| **Epochs** | `50-100` | **"Repetitions".** How many times the AI reads the entire textbook. <br>• *Too few*: It doesn't learn. <br>• *Too many*: It memorizes instead of understanding ("Overfitting"). |
| **Learning Rate** | `0.001` | **"Step Size".** How fast the AI changes its mind. <br>• **High**: Learns fast but might overshoot the answer. <br>• **Low**: Precise but takes forever. |
| **Num Workers** | `4-8` | **"Helpers".** CPU cores fetching data for the GPU. <br>• Set to half your CPU cores. Too high = computer freezes. |
| **Architecture** | `ResNet18` | **"Brain Type".** <br>• **`ResNet18`**: Smart, heavy, accurate. Best for servers. <br>• **`MobileNet`**: Fast, lightweight. Best for Phones/Pi. <br>• **`TinyConv`**: Ultra-tiny. Best for ESP32/Arduino. |

## 🛠️ Advanced Training Options

| Variable | What it is |
| :--- | :--- |
| **Early Stopping** | "The Quitter". Stops training if the AI stops improving for X rounds. Prevents wasting electricity. |
| **Focal Loss** | "Focus on the Hard Stuff". Forces the AI to care more about the 1% of hard files than the 99% of easy ones. |
| **Distillation** | "Teacher-Student". Trains a small (student) model to mimic a huge (teacher) model. <br>• **Teacher**: Wav2Vec2 (Huge/Smart). <br>• **Student**: Your TinyConv model. |
| **Include Mined** | If checked, adds the "Hard Negatives" you found in previous tests. **Crucial** for fixing false alarms. |
| **Use EMA** | "Smoothing". Averages the model weights over time. Results in a more stable final model. |
| **Time/Pitch Shift** | "Augmentation". Randomly speeds up or changes pitch of audio during training. Makes the AI robust to different speakers. |

                    """
                )

            # ==========================================================================================================
            # 3. TRAINING & EVALUATION
            # ==========================================================================================================
            with gr.TabItem("📈 Training & Evaluation"):
                gr.Markdown(
                    """
# 📈 Understanding Training & Evaluation (Panels 3 & 4)

## 📊 The Metrics (What the numbers mean)

When training starts, you will see a lot of numbers. Here is how to read them:

### 1. Loss (The "Badness" Score) 📉
- **Goal**: You want this **LOW**.
- **Train Loss**: How many mistakes it makes on the study guide.
- **Val Loss**: How many mistakes it makes on the practice exam.
- **Healthy**: Both go down together.
- **Overfitting**: Train Loss goes DOWN, but Val Loss goes UP. (The AI is cheating/memorizing).

### 2. Accuracy 🎯
- **Goal**: You want this **HIGH** (close to 100%).
- **Warning**: If you have 99 negatives and 1 positive, a model that says "Negative" to everything has 99% accuracy but is USELESS. Do not trust accuracy alone.

### 3. FPR (False Positive Rate) - The "Annoyance" Metric 🚨
- **What it refers to**: How often the AI wakes up when you didn't call it.
- **Goal**: **0.00%** (Zero).
- **Reality**: Anything under 1% is good. Over 5% is unusable (it will wake up every time you cough).

### 4. FNR (False Negative Rate) - The "Deafness" Metric 🧏
- **What it refers to**: How often the AI ignores you when you DO call it.
- **Goal**: Low (under 5%).
- **Trade-off**: Lowering FPR usually raises FNR. You must find a balance.

---

## 🎯 Model Evaluation (Panel 4)

Once training is done, you verify the model here.

### 🧪 1. File Evaluation
Upload specific `.wav` files to test.
- **Use case**: Upload a file of you saying "Hey Computer" and a file of you saying "Hey Tomato". It should detect the first and ignore the second.

### 🎤 2. Microphone Test
Live testing. speak to your computer.
- **Threshold Slider**: The "Confidence Bar".
  - **High (0.8+)**: Hard to wake up, but rarely makes mistakes.
  - **Low (0.3)**: Wakes up easily, but might wake up to random noise.

### 📉 3. Test Set Evaluation (The Final Exam)
Runs the model against the "Test Split" (the data it has never seen).
- **Confusion Matrix**: A box showing:
  - **TP (True Positive)**: Correctly Woke Up.
  - **TN (True Negative)**: Correctly Stayed Asleep.
  - **FP (False Positive)**: Woke up incorrectly (Bad!).
  - **FN (False Negative)**: Stayed asleep incorrectly (Bad!).

### ⛏️ 4. Background Mining (Sentry Mode)
- **What is it?**: Runs the model against hours of TV/Radio/Noise.
- **Why?**: To find "False Positives". If it wakes up to a car horn, we save that sound as a "Hard Negative" and re-train the model to fix it.

                    """
                )

            # ==========================================================================================================
            # 4. EXPORT & DEPLOYMENT
            # ==========================================================================================================
            with gr.TabItem("🚀 Export & Deployment"):
                gr.Markdown(
                    """
# 🚀 Export & Deployment (Panel 5)

Your model is trained (PyTorch `.pt` file). Now you need to make it run on your actual device.

## formats

### 📦 ONNX (Open Neural Network Exchange)
- **What it is**: The universal adapter. Runs on Windows, Linux, Mac, Python, C++.
- **Opset Config**: Version of the ONNX standard. `14` is safe.
- **Dynamic Batch**: Allows sending 1 file or 100 files at once. Keep this checked.

### 📱 TFLite (TensorFlow Lite)
- **Target**: **Microcontrollers** (ESP32, Raspberry Pi Zero, Android).
- **Requirement**: You must enable this if you want to run on ESPHome.

## 🗜️ Quantization (Shrinking the Brain)

Models are normally `FP32` (32-bit decimal numbers, e.g., 3.14159...). Devices like ESP32 are slow at math with big numbers.

### 1. FP16 (Halving)
- **Effect**: 50% smaller size. 2x faster on some GPUs.
- **Accuracy Loss**: Almost zero.
- **Recommended**: For PC/Phone deployment.

### 2. INT8 (Quartering) / QAT
- **Effect**: 75% smaller size. 4x faster.
- **What it does**: Rounds numbers to integers (0-255).
- **Accuracy Loss**: noticeable unless you trained with **QAT**.
- **QAT (Quantization Aware Training)**: A training mode (Panel 2) where the AI "practices" being an integer model.
- **Recommended**: For **ESP32 / ESPHome**.

## 🏠 ESPHome Information

- **Compatible Model**: `TinyConv` or `MobileNetV3`. (ResNet18 is too big for ESP32).
- **Steps**:
  1. Train a `TinyConv` model.
  2. Go to Export Panel.
  3. Check **Export TFLite** AND **INT8 Quantization**.
  4. Check **ESPHome Compatibility**.
  5. The file will be saved to `exports/esphome/`.
  6. Copy this file to your ESPHome configuration.

                    """
                )

            # ==========================================================================================================
            # 5. TROUBLESHOOTING
            # ==========================================================================================================
            with gr.TabItem("🔧 Troubleshooting"):
                gr.Markdown(
                    """
# 🔧 Troubleshooting Guide

## ❌ "CUDA / GPU Out of Memory"
**Translation**: Your graphics card brain is full.
**Fixes**:
1. **Lower Batch Size** (Panel 2): Try 32 → 16 → 8.
2. **Restart App**: Sometimes memory gets stuck.
3. **Turn off Distillation**: The "Teacher" model is huge.

## ❌ "Loss is NaN" (Not a Number)
**Translation**: The AI broke. It tried to learn too fast and the numbers exploded to infinity.
**Fixes**:
1. **Lower Learning Rate**: Try adding a zero (0.001 → 0.0001).
2. **Check Data**: A single corrupted audio file can cause this. Run "Scan Datasets" again.

## ❌ "It wakes up to everything!" (High False Positive)
**Translation**: The model is trigger-happy.
**Fixes**:
1. **Hard Negatives**: You need more files in the `hard_negative/` folder.
2. **Background Noise**: Increase "Background Noise Probability" in Config.
3. **Threshold**: Raise the detection threshold (0.5 → 0.8).

## ❌ "It never wakes up!" (High False Negative)
**Translation**: The model is deaf or too shy.
**Fixes**:
1. **More Positives**: You didn't provide enough examples of the wakeword.
2. **Variety**: Record yourself whispering, shouting, faster, slower.
3. **Focal Loss**: Enable this in Config (Advanced) to force learning of the wake word.

---
**Still stuck?** Check the terminal/console window for detailed error logs (red text).
                    """
                )

    return panel
