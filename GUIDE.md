# 🎓 The Ultimate Wakeword Masterclass: From Sound to Silicon

Welcome to the definitive guide for this project. Whether you're a beginner watching numbers move for the first time or an engineer tuning a production model, this guide explains **every algorithm, every config, and every metric** using the logic of the "Wakeword Universe."

---

## 🌌 1. The Big Picture: How it All Works
Building a wakeword model is like teaching a guard dog. You want it to sleep through the rain (noise) but wake up instantly when it hears its name (wakeword), and **never** bark at a random squirrel (false alarm).

### The Lifecycle:
1.  **Data**: Collect raw sounds.
2.  **Features**: Turn sounds into "pictures" (Spectrograms).
3.  **Training**: Show pictures to the model and correct its guesses.
4.  **Optimization**: Make the model smaller and smarter.
5.  **Deployment**: Put the model on a tiny chip to listen 24/7.

---

## 📂 2. Foundations: Data & "The Painting"

### 🔍 Scans & Splits
*   **The Scan**: We look through your folders and find every `.wav` file.
*   **The Split (80/10/10)**: We divide data into three buckets:
    *   **Train**: The "Study Material."
    *   **Val (Validation)**: The "Practice Quiz" (shown during training).
    *   **Test**: The "Final Exam" (never seen until the very end).

### 🎨 Feature Extraction (Spectrograms)
Computers can't "hear," but they are great at seeing. We turn audio into a **Spectrogram**—a heat map of frequency over time.
*   **Mel Spectrogram**: This uses the "Mel Scale," which mimics how human ears hear (we are better at telling apart low sounds than high sounds).
*   **MFCC**: A compressed version of the Mel Spectrogram. It's like a low-resolution thumbnail. Great for very weak chips (ESP32).
*   **CMVN (The Microphone Neutralizer)**: Every microphone has a "tint." One might be bassy, another tinny. CMVN (Cepstral Mean and Variance Normalization) subtracts the "average tint" so the model sees a neutral version of the sound regardless of the hardware.

---

## 🏋️ 3. The Training Gym: Algorithms & Configs

### 🔄 The Training Loop Step-by-Step
1.  **Load Batch**: Grab 64 files.
2.  **Augment**: "Ruin" the files slightly (Pitch, Noise, Reverb).
3.  **Forward Pass**: Model makes a guess (0 to 1).
4.  **Compute Loss**: How far was the guess from the truth?
5.  **Backward Pass**: Calculate which "brain cells" (weights) were wrong.
6.  **Optimizer Step**: Use **AdamW** (the "Coach") to nudge those weights.

### 🌪️ Augmentation (The Boot Camp)
We don't want a "fair-weather" model. We want a survivor.
*   **Pitch Shift**: Changes the voice from a squeaky child to a deep adult.
*   **Time Stretch**: Handles people who talk like auctioneers or sloths.
*   **RIR (Room Impulse Response)**: The "Echo" simulator. It makes the sound bounce like it's in a tiled bathroom or a cavernous hall.
*   **SpecAugment (The Blindfold)**: We cut random black stripes into the Spectrogram. If the model can still guess "Hey Assistant" while 20% of the image is missing, it truly understands the *structure* of the word, not just the pixels.

### 🛡️ EMA (The Wise Elder)
During training, the model's brain is changing rapidly—sometimes it gets "over-excited" by a lucky batch. **Exponential Moving Average (EMA)** keeps a separate version of the model that changes slowly. It's like having a wise old version of the model that only listens to the *best* and most consistent changes. **This version is usually 5-10% more accurate.**

### ⚡ Mixed Precision (AMP)
Normally, computers use 32-bit "High Detail" numbers. **AMP** switches the easy math to 16-bit "Medium Detail." It's like doing math on a napkin instead of a graph paper—it's **2-3x faster** and uses half the memory, with zero loss in quality.

---

## 📊 4. The Scoreboard: Metrics & Their "Why"

### 📉 Loss vs. Accuracy
*   **Loss**: The "Distance to Perfection." If loss is 0.5, the model is confused. If it's 0.01, the model is a genius.
*   **Accuracy**: The percentage of correct guesses. **Warning**: If you have 99% noise and 1% wakeword, a model that says "Noise" every time is 99% accurate but totally useless! This is why we need...

### ⚖️ FPR vs. FNR (The Eternal War)
*   **FPR (False Positive Rate)**: The "False Alarm." Model triggers when you didn't say the word. (Annoying).
*   **FNR (False Negative Rate)**: The "Missed Trigger." You said the word and it ignored you. (Frustrating).
*   **The Trade-off**: If you make the model "easier to trigger," FPR goes up. If you make it "stricter," FNR goes up.

### 🏆 FAH (False Alarms per Hour) - The King of Metrics
In the real world, accuracy doesn't matter as much as: **"How many times will this thing wake me up at night for no reason?"** FAH tells you exactly that. A production-grade model should have **<1 FAH**.

### 📉 EER & pAUC
*   **EER (Equal Error Rate)**: The point where FPR and FNR are exactly the same. It's the "Fairness Score."
*   **pAUC (Partial AUC)**: We only care about the model's performance when the False Alarm rate is **very low** (under 10%). pAUC measures the "Area under the curve" in that high-stakes zone.

---

## 💎 5. The Master Strokes: Optimization

### 🎓 Knowledge Distillation (Master/Apprentice)
We take a giant, genius model (The **Teacher**, like Wav2Vec2) and have it "tutor" a tiny model (The **Student**, like MobileNet). 
*   **The Secret**: Instead of the student just seeing "Right/Wrong," the teacher shows its "Soft Probabilities"—meaning it shows the student *why* it thought a sound was almost a word. This "transfers the brain" of the giant model into the tiny one.

### 💎 QAT (Quantization Aware Training)
Normal model weights are decimals (0.12345). Tiny chips (ESP32) hate decimals. **QAT** trains the model using only whole numbers (Integers). It’s like teaching a painter to work with only 256 colors instead of millions. The result is a model that is **4x smaller and 3x faster.**

---

## 🏰 6. Deployment: The "Distributed Cascade"

### 🛡️ Sentry vs. ⚖️ Judge
We use a two-stage defense:
1.  **The Sentry (The Satellite)**: A tiny, ultra-fast model (MobileNet/TinyConv) living on your ESP32. It listens 24/7. It's allowed to be a bit "jumpy."
2.  **The Judge (The Server)**: When the Sentry thinks it heard the word, it wakes up the **Judge** (a big ResNet/Conformer on your server). The Judge does the heavy math and gives the final "Yes" or "No."
*   **Result**: You get Server-grade accuracy with 0.1W of power usage.

---

## 🔧 7. The Mechanic's Handbook: Troubleshooting

| What you see... | What it means... | What to do... |
| :--- | :--- | :--- |
| **Loss is `NaN`** | The "Knobs" broke. Math exploded. | Lower your `Learning Rate` or check for "corrupt" audio files. |
| **Accuracy is 100% in Epoch 1** | The model "Cheated." | You likely have the same files in both Train and Val folders. Check your splits! |
| **FPR is high, FNR is low** | The dog barks at everything. | Increase `hard_negative_weight` or use a higher `Hysteresis High` threshold. |
| **Speed is 0 samples/sec** | The data loader is stuck. | Check if your `num_workers` is too high for your CPU. Try 4 or 8. |
| **ETA keeps jumping** | Training speed is unstable. | Enable `NPY Caching` to make data loading consistent. |

---

## 🌟 8. The Golden Ratios (Best Practices)

1.  **The Balanced Diet**: Your dataset should be roughly 1:5 (1 Positive for every 5 Negatives). Too many positives makes the model "lazy."
2.  **The Warmup**: Always use 5 epochs of **LR Warmup**. It prevents the model from "panicking" when it first sees the data.
3.  **The Context**: A 1.5s window is the "Sweet Spot." Too short and you miss the start/end; too long and the model gets distracted by silence.
4.  **The HPO Secret**: Use **Optuna** to find the best Learning Rate. Don't guess—let the math decide.

---

## ⚪ 9. Phase 6: Deployment & Streaming (The "Real World")
Once the model is trained, it needs to live on a device and listen 24/7.

### 📡 Streaming Settings (`StreamingConfig`)
*   **Hysteresis (High: 0.7, Low: 0.3)**: This is the "Decision Buffer." 
    *   To trigger the word, the model must be **70% sure**. 
    *   To "reset" and start listening again, the score must drop below **30%**. 
    *   This prevents the model from "stuttering" (triggering 5 times for one word).
*   **Smoothing Window (5)**: The model doesn't just look at one split-second; it averages its last 5 guesses. This filters out random "blips" of noise.
*   **Cooldown (500ms)**: After hearing the word, the model takes a 0.5-second "nap" so it doesn't hear itself echo.

### 📏 Constraints (`SizeTargetConfig`)
*   **Max Flash/RAM**: If you are deploying to a tiny chip (like an ESP32), you can set a limit (e.g., 200KB). The trainer will warn you if your model is getting too "fat" for its home.

---

## 🧩 10. The Strategy Guide: "Why are we the way we are?"

This final section explains our default values and how you should evolve them as your project grows.

### 🧠 Our "Default" Philosophy
Our default values (e.g., **Learning Rate 5e-4**, **Batch Size 64**) are chosen as the "Safe Middle Ground." They are designed to work on a 30-series NVIDIA GPU with a dataset of about 5,000–20,000 samples. 

### 📈 Scaling Up: When to change what?

| If you change... | ...then you must adjust: | Why? |
| :--- | :--- | :--- |
| **Dataset Size** (Going from 10h to 100h) | **Increase Epochs** and **Batch Size**. | Larger datasets need more time to see everything, but you can process more at once to save time. |
| **Model Size** (Switching to ResNet18) | **Lower Learning Rate** (e.g., to 1e-4). | Big models are "delicate." A high learning rate will make them "trip over" and explode (NaN loss). |
| **Edge Hardware** (Moving to ESP32) | **Decrease Mel Bands** (40) and **n_fft**. | ESP32 has very little RAM. Every Mel Band you add eats memory. 40 is the sweet spot for efficiency. |
| **Environment** (Noisy Factory vs. Quiet Lab) | **Increase RIR Probability** and **Noise SNR**. | A model trained in a lab will be "blind" in a factory. You need to force it to learn through the chaos. |

### 🛠️ How to "Level Up" your Model
If you want to move from a 90% accuracy prototype to a 99.9% production-ready system, here is your roadmap:

1.  **Feed the Beast (Data)**: A model is only as good as what it hears.
    *   **Diversity**: Add voices with different accents, ages, and genders.
    *   **Hard Negatives**: Add audio of people saying words *similar* to your wakeword (e.g., "Hey Assistant" vs "Hey Assistant's"). This is the #1 way to kill False Alarms.
2.  **The Distillation Cost**: Using Knowledge Distillation is **expensive** in training time (takes 2-3x longer) but the "Cost" is worth it for the "Reward": a tiny model that performs like a giant.
3.  **The QAT Trade-off**: Quantization (QAT) makes the model faster, but it always loses a *tiny* bit of accuracy (usually <1%). Only use it if your hardware requires it (MCU/Mobile).
4.  **The "Human" Metric**: Don't just watch the F1 score. Watch the **Latency**. A model that is 100% accurate but takes 2 seconds to respond feels "broken" to a user. Aim for **<200ms** total latency.

---
**The Goal**: We are aiming for **"Transparent Technology."** The best wakeword system is one the user forgets is even there. Keep turning the knobs, watching the FAH, and listening to the feedback! 🚀🎧
