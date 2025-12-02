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
            "The complete, definitive guide to the Wakeword Training Platform. This documentation is designed to take you from a beginner to an expert in training production-grade voice models."
        )

        with gr.Tabs():
            # -------------------------------------------------------------------------
            # 1. Introduction & Architecture
            # -------------------------------------------------------------------------
            with gr.TabItem("🏠 Introduction"):
                gr.Markdown(
                    """
# Welcome to the Wakeword Training Platform

## What is this Platform?
This is not just a training script; it is a **production-ready ecosystem** for creating custom wakeword detection models (like "Hey Siri", "Alexa", or "Okay Google"). It is built to solve the specific challenges of deploying voice AI in the real world, such as battery life constraints, noisy environments, and the need for extremely low false alarm rates.

## The "Google-Tier" Distributed Cascade Architecture
This platform supports a sophisticated 3-stage architecture designed to balance accuracy and power consumption:

1.  **Stage 1: The Sentry (Edge Device)**
    *   **Role**: Always-on listening.
    *   **Model**: Ultra-lightweight (e.g., `tiny_conv` or `mobilenetv3`).
    *   **Goal**: Wake up only when it thinks it hears the keyword. It is allowed to have some false alarms to ensure it never misses a real command.
    *   **Deployment**: Runs on microcontrollers like ESP32 or DSPs.

2.  **Stage 2: The Judge (Local Server/Hub)**
    *   **Role**: Verification.
    *   **Model**: Heavy and accurate (e.g., `resnet18` or `wav2vec2`).
    *   **Goal**: Analyze the audio clip triggered by the Sentry and decide if it was a real wakeword or just a similar sound (e.g., "Alexa" vs. "Election").
    *   **Deployment**: Runs on a Raspberry Pi, Home Assistant server, or local PC.

3.  **Stage 3: The Teacher (Training Cluster)**
    *   **Role**: Knowledge Distillation.
    *   **Model**: Massive pre-trained models.
    *   **Goal**: Teach the smaller models how to behave by transferring knowledge, improving their accuracy beyond what they could learn on their own.

## Core Capabilities
- **GPU Acceleration**: Fully utilizes NVIDIA GPUs with CUDA for rapid training.
- **Mixed Precision (AMP)**: Uses FP16 precision to double training speed and halve memory usage without losing accuracy.
- **Smart Data Pipeline**: Automatically balances your dataset, normalizes audio volume, and applies "On-the-fly" augmentation to create infinite variations of your data.
- **Production Metrics**: We don't just show "Accuracy". We track **False Alarms per Hour (FAH)** and **Equal Error Rate (EER)**, which are the industry standards for shipping voice products.

---
*Use the tabs above to navigate through the detailed guides.*
                    """
                )

            # -------------------------------------------------------------------------
            # 2. Detailed Configuration Guide
            # -------------------------------------------------------------------------
            with gr.TabItem("📘 Configuration Guide"):
                gr.Markdown("# Detailed Configuration Guide")
                gr.Markdown("Every parameter in this platform has a specific purpose. Here is a deep dive into what they do and how to choose the right values.")
                
                with gr.Accordion("1. Data Configuration (The Foundation)", open=True):
                    gr.Markdown(
                        """
The quality of your model depends entirely on how you process the audio data.

### `sample_rate` (Default: 16000)
*   **What it is**: The number of audio snapshots taken per second (Hz).
*   **Why 16000?**: Human speech intelligibility is mostly contained below 8kHz. According to the Nyquist theorem, a 16kHz sample rate captures frequencies up to 8kHz, which is perfect for speech.
*   **When to change**:
    *   **Use 8000**: Only for extremely low-power chips where every byte counts. Quality will suffer.
    *   **Use 44100/48000**: Generally unnecessary for wakewords and will just slow down training and inference by 3x.

### `audio_duration` (Default: 1.0 - 1.5s)
*   **What it is**: The fixed length of the audio window the model looks at.
*   **Best Practice**: Your wakeword should fit comfortably within this window.
    *   Short words ("Alexa"): **1.0s** is usually enough.
    *   Long phrases ("Hey Google"): **1.5s** or **2.0s** might be needed.
*   **Trade-off**: Longer duration = More context for the model, but higher latency (the user has to wait longer for a response).

### `feature_type` (Default: "mel")
*   **What it is**: How raw audio waves are converted into an image (spectrogram) for the AI.
*   **Recommendation**: Always use **"mel"** (Mel Spectrogram). It mimics how the human ear perceives sound (more sensitivity to lower frequencies).
*   **`n_mels`**: The vertical resolution of the spectrogram.
    *   **64**: The gold standard. Good balance of detail and speed.
    *   **40**: Use this for edge devices (ESP32) to reduce computation.
    *   **80+**: Overkill for simple wakewords, mostly used for complex speech recognition (ASR).

### `normalize_audio` (Default: True)
*   **What it is**: Adjusts the volume of every clip to a standard level.
*   **Why it matters**: You don't want your model to think that "Loud" = "Wakeword". It should trigger even if you whisper. This setting ensures volume invariance.
                        """
                    )

                with gr.Accordion("2. Model Architecture (The Brain)", open=False):
                    gr.Markdown(
                        """
Choosing the right brain for your application is critical.

### `resnet18` (The Powerhouse)
*   **Description**: A deep convolutional network with "residual connections" that allow it to learn very complex patterns without getting stuck.
*   **Best For**: **Server-side verification ("The Judge")** or powerful edge devices (Raspberry Pi 4, Jetson Nano).
*   **Pros**: Highest accuracy, very robust to noise.
*   **Cons**: Large file size (~45MB), slower inference.

### `mobilenetv3` (The Balanced Choice)
*   **Description**: Designed specifically for mobile phones. Uses "depthwise separable convolutions" to reduce math operations by ~8x.
*   **Best For**: **Smartphones, High-end Microcontrollers**.
*   **Pros**: Fast, lightweight (~3-5MB), good accuracy.
*   **Cons**: Slightly less robust than ResNet in very noisy environments.

### `tiny_conv` (The Minimalist)
*   **Description**: A custom, extremely shallow network.
*   **Best For**: **Low-power Microcontrollers (ESP32, Arduino)**.
*   **Pros**: Tiny (<100KB), ultra-fast, low battery usage.
*   **Cons**: Lower accuracy, higher false alarm rate. Intended to be used as a "Sentry" that wakes up a bigger processor.

### `dropout` (Default: 0.2 - 0.5)
*   **What it is**: During training, we randomly "turn off" a percentage of neurons.
*   **Why**: This forces the model to not rely on any single feature (like a specific frequency), making it more robust.
*   **Tuning**:
    *   **0.2**: Good for large datasets.
    *   **0.5**: Use if you have a small dataset to prevent overfitting (memorization).
                        """
                    )

                with gr.Accordion("3. Training Parameters (The Process)", open=False):
                    gr.Markdown(
                        """
### `batch_size` (Default: 32 - 128)
*   **What it is**: How many audio clips the model studies at once before updating its brain.
*   **Trade-off**:
    *   **Larger (64, 128)**: Faster training, more stable gradient estimates (smoother learning). Requires more GPU VRAM.
    *   **Smaller (8, 16)**: Uses less memory, but training can be "noisy" and erratic.
*   **Recommendation**: Set this as high as your GPU allows without crashing (OOM).

### `learning_rate` (Default: 0.001)
*   **What it is**: The "step size" the model takes when trying to improve.
*   **Analogy**: Imagine hiking down a mountain in the fog.
    *   **Too High**: You take giant leaps and might jump over the valley (fail to converge).
    *   **Too Low**: You take tiny baby steps and it takes forever to reach the bottom.
*   **Tip**: Use the **Learning Rate Finder** feature in the code to automatically find the perfect starting value.

### `epochs` (Default: 50+)
*   **What it is**: One full pass through your entire dataset.
*   **Strategy**: Don't worry about setting this too high. We use **Early Stopping**, which automatically kills the training if the model stops improving for `early_stopping_patience` epochs.

### `optimizer` (Default: AdamW)
*   **Recommendation**: Stick with **AdamW**. It is the modern standard. It combines the adaptive learning of Adam with better weight decay (regularization) handling, leading to models that generalize better to new data.
                        """
                    )

                with gr.Accordion("4. Augmentation (The Secret Sauce)", open=False):
                    gr.Markdown(
                        """
Real-world data is messy. Augmentation simulates this messiness so your model isn't surprised when deployed.

### `background_noise_prob` (Critical!)
*   **What it is**: The probability of mixing a clean voice sample with background noise (cafe, rain, traffic).
*   **Recommendation**: Set to **0.5 - 0.7**.
*   **Why**: Without this, your model will work perfectly in a quiet room but fail immediately if a TV is on.

### `rir_prob` (Room Impulse Response)
*   **What it is**: Simulates the echo/reverb of different rooms (bathroom, living room, hall).
*   **Why**: A voice sounds very different in a tiled bathroom vs. a carpeted bedroom. This helps the model ignore those acoustic differences.

### `time_stretch` & `pitch_shift`
*   **What it is**: Randomly speeding up/slowing down audio and changing the pitch.
*   **Why**: This simulates different speakers (fast talkers, slow talkers, high/low voices) without needing to actually record 1000 different people.
                        """
                    )

            # -------------------------------------------------------------------------
            # 3. Training & Metrics Deep Dive
            # -------------------------------------------------------------------------
            with gr.TabItem("📊 Metrics & Analysis"):
                gr.Markdown("# Understanding Training Metrics")
                gr.Markdown("How to read the numbers and diagnose your model's health.")

                with gr.Accordion("The 'Big Three' Metrics", open=True):
                    gr.Markdown(
                        """
### 1. F1 Score (The King of Metrics)
*   **Definition**: The harmonic mean of Precision and Recall.
*   **Why it matters**: Accuracy is useless. If you have 990 negative samples and 10 positive samples, a model that predicts "Negative" for everything has 99% accuracy but is useless. F1 Score penalizes this.
*   **Interpretation**:
    *   **> 0.90**: Excellent. Production-ready.
    *   **0.80 - 0.90**: Good. Usable but might need a secondary verification stage.
    *   **< 0.50**: Poor. The model is confused or the data is bad.

### 2. FPR (False Positive Rate) - "The Annoyance Factor"
*   **Definition**: Out of all the times the world was silent or people were talking about other things, what percentage of time did the model scream "I heard it!"?
*   **Target**: You want this as close to **0.00%** as possible.
*   **Real-world impact**: If FPR is 1%, and your device listens to 1000 windows per hour, it will wake up randomly 10 times an hour. That is unacceptable. **Target < 0.1%**.

### 3. FNR (False Negative Rate) - "The Frustration Factor"
*   **Definition**: Out of all the times you actually said the wakeword, what percentage of time did the model ignore you?
*   **Target**: **< 5%**.
*   **Real-world impact**: If FNR is high, users have to repeat themselves ("Alexa... ALEXA!"). This leads to user frustration and product abandonment.
                        """
                    )

                with gr.Accordion("Diagnosing Common Problems", open=True):
                    gr.Markdown(
                        """
### Scenario A: "The Overfit"
*   **Symptoms**: Training Loss goes DOWN, but Validation Loss goes UP. Training Accuracy is 99%, Validation Accuracy is 80%.
*   **Diagnosis**: The model is memorizing the training files instead of learning the sound of the word.
*   **Solution**:
    1.  Increase **Dropout**.
    2.  Increase **Weight Decay**.
    3.  Add more **Augmentation** (Noise, RIR).
    4.  Get more diverse training data.

### Scenario B: "The Trigger Happy"
*   **Symptoms**: High Recall (catches the wakeword), but very high False Positive Rate.
*   **Diagnosis**: The model thinks *everything* is the wakeword. It hasn't learned to discriminate.
*   **Solution**:
    1.  Add **Hard Negatives**: Find words that sound like your wakeword (e.g., for "Marvin", record "Carving", "Starving") and add them to the negative dataset.
    2.  Increase `hard_negative_weight` in the config.
    3.  Increase the ratio of negative samples in the dataset.

### Scenario C: "The Deaf Model"
*   **Symptoms**: Loss doesn't decrease. F1 Score stays near 0.
*   **Diagnosis**: The model isn't learning anything.
*   **Solution**:
    1.  Check your data! Are the labels correct? Is the audio silent?
    2.  Your **Learning Rate** might be too high (exploding) or too low (stuck).
    3.  Try "Overfitting a single batch": Train on just 10 samples. If it can't learn those perfectly, your code/pipeline is broken.
                        """
                    )

            # -------------------------------------------------------------------------
            # 4. Advanced Technical Reference
            # -------------------------------------------------------------------------
            with gr.TabItem("⚙️ Technical Deep Dive"):
                gr.Markdown("# Under the Hood")
                gr.Markdown("Detailed explanation of the advanced algorithms powering this platform.")

                with gr.Accordion("1. CMVN (Cepstral Mean and Variance Normalization)", open=False):
                    gr.Markdown(
                        """
**The Problem**: A recording from a high-end studio microphone has very different statistical properties (mean energy, variance) compared to a recording from a cheap laptop mic. A model trained on one might fail on the other.

**The Solution (CMVN)**:
Before the model sees the spectrogram, we calculate the statistical Mean and Variance of the entire dataset. We then subtract the Mean and divide by the Variance for every single pixel.
*   **Result**: All audio, regardless of source, is transformed into a "standard normal distribution" (Mean=0, Std=1).
*   **Benefit**: The model stops caring about the microphone quality or volume and focuses purely on the *shape* of the sound (phonemes). This typically boosts accuracy by 2-4%.
                        """
                    )

                with gr.Accordion("2. Balanced Batch Sampling", open=False):
                    gr.Markdown(
                        """
**The Problem**: In wakeword detection, you usually have 1,000 positive samples and 50,000 negative samples (background noise, speech). If you train naively, 98% of every batch will be "Negative". The model will just learn to always guess "Negative" and achieve 98% accuracy without learning anything.

**The Solution**:
We use a custom `WeightedRandomSampler` that forces every training batch to have a specific ratio, regardless of the total dataset size.
*   **Default Ratio**: 1:1:1 (Positive : Negative : Hard Negative).
*   **Benefit**: The model sees the wakeword just as often as it sees silence, forcing it to learn the difference. This is crucial for convergence.
                        """
                    )

                with gr.Accordion("3. Quantization Aware Training (QAT)", open=False):
                    gr.Markdown(
                        """
**The Problem**: Standard AI models use 32-bit floating point numbers (FP32). Microcontrollers (like ESP32) are very slow at FP32 math but very fast at 8-bit integer (INT8) math. Converting a model to INT8 after training usually destroys accuracy.

**The Solution (QAT)**:
We simulate the rounding errors of INT8 *during training*. The model learns to adapt its weights to be robust to this loss of precision.
*   **Workflow**:
    1.  Train normally for a few epochs.
    2.  Enable QAT. The system inserts "Fake Quantization" nodes into the network.
    3.  Continue training. The model "heals" itself from the quantization damage.
    4.  Export. The resulting model is ready for INT8 deployment with almost zero accuracy loss.
                        """
                    )

                with gr.Accordion("4. Knowledge Distillation", open=False):
                    gr.Markdown(
                        """
**The Concept**: How do you make a tiny model (Student) as smart as a huge model (Teacher)?

**The Process**:
1.  We take a massive, pre-trained model (like Wav2Vec 2.0) that understands all human speech perfectly. This is the **Teacher**.
2.  We have our tiny MobileNet model. This is the **Student**.
3.  We pass the same audio to both.
4.  The Student tries to predict the label (Wakeword/Not Wakeword).
5.  **Crucially**, the Student is also punished if its internal activation patterns don't match the Teacher's patterns.
6.  **Result**: The Student learns to "think" like the Teacher, achieving higher accuracy than it could ever reach on its own.
                        """
                    )

            # -------------------------------------------------------------------------
            # 5. Troubleshooting & FAQ
            # -------------------------------------------------------------------------
            with gr.TabItem("🔧 Troubleshooting"):
                gr.Markdown("# Troubleshooting Guide")
                
                with gr.Accordion("Installation & Environment", open=True):
                    gr.Markdown(
                        """
### "CUDA not available" / Training on CPU
*   **Cause**: PyTorch cannot see your NVIDIA GPU.
*   **Solution**:
    1.  Open a terminal.
    2.  Run `nvidia-smi`. If this fails, install NVIDIA Drivers.
    3.  Reinstall PyTorch with the correct CUDA version:
        `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### "ImportError: dll load failed" (Windows)
*   **Cause**: Missing C++ redistributables or corrupted environment.
*   **Solution**: Install the "Microsoft Visual C++ Redistributable" and recreate your Conda/Python environment.
                        """
                    )

                with gr.Accordion("Training Errors", open=True):
                    gr.Markdown(
                        """
### "CUDA Out of Memory" (OOM)
*   **Cause**: The batch size is too large for your GPU's VRAM.
*   **Solutions (in order)**:
    1.  Reduce `batch_size` (e.g., 64 -> 32 -> 16).
    2.  Enable `mixed_precision` in the config (uses 50% less memory).
    3.  Reduce `audio_duration` (e.g., 1.5s -> 1.0s).
    4.  Switch to a smaller model architecture (`mobilenetv3`).

### Loss is NaN (Not a Number)
*   **Cause**: The math exploded. Usually because the Learning Rate is too high.
*   **Solution**:
    1.  Reduce `learning_rate` by 10x (e.g., 0.001 -> 0.0001).
    2.  Check your dataset for corrupted audio files (0-byte files or pure static).
    3.  Enable `gradient_clipping` in the config.
                        """
                    )

                with gr.Accordion("Deployment Issues", open=True):
                    gr.Markdown(
                        """
### "Model works on PC but fails on ESP32"
*   **Cause**: Domain mismatch or quantization loss.
*   **Solution**:
    1.  Did you use **QAT**? If not, the INT8 conversion probably destroyed the accuracy.
    2.  **Microphone Mismatch**: The ESP32 mic is likely much worse than your PC mic. You need to record training data *using the ESP32 mic* or apply heavy augmentation (Bandpass filter) to simulate it.

### "Too many false alarms in the living room"
*   **Cause**: The model wasn't trained on "TV noise".
*   **Solution**:
    1.  Download a "TV/Movie Audio" dataset.
    2.  Add it to your `background_noise` folder.
    3.  Retrain. The model needs to learn that "people talking on TV" != "Wakeword".
                        """
                    )

        gr.Markdown("---")
        gr.Markdown(
            "*This documentation is automatically generated and updated based on the latest system capabilities.*"
        )

    return panel


if __name__ == "__main__":
    # Test the panel
    demo = create_docs_panel()
    demo.launch()
