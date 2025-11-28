# Wakeword Training Guide: Understanding the Metrics

This guide explains the technical terms you see during training in simple language. It tells you what the numbers mean, what counts as a "good" result, and how to spot problems early.

## 1. The "Big Three" Metrics to Watch

These are the most important numbers to check to see if your model is actually learning.

### ✅ F1 Score (The King of Metrics)
*   **What it is:** The single best number to judge your model. It balances "catching the wakeword" vs. "ignoring noise."
*   **What is good?**
    *   **> 0.90:** Excellent. Production-ready.
    *   **0.80 - 0.90:** Good. Usable but might make occasional mistakes.
    *   **< 0.50:** Poor. The model is confused.
    *   **0.00:** Failed. The model is either sleeping (predicting nothing) or panicking (predicting everything).

### ❌ FPR (False Positive Rate) - " The Annoyance Factor"
*   **What it is:** How often the model activates when you *didn't* say the wakeword.
*   **What is good?** **Lower is better.**
    *   **0.00% - 0.50%:** Excellent. Very rarely interrupts you.
    *   **> 5%:** Terrible. It will wake up constantly from random noise.

### ❌ FNR (False Negative Rate) - "The Frustration Factor"
*   **What it is:** How often the model *ignores* you when you actually say the wakeword.
*   **What is good?** **Lower is better.**
    *   **< 5%:** Excellent. Hears you almost every time.
    *   **> 20%:** Frustrating. You have to shout or repeat yourself.

---

## 2. Secondary Metrics (The Details)

### Accuracy
*   **What it is:** Percentage of total correct predictions.
*   **The Trap:** **Ignore this.** If your dataset is 90% negative audio, a dumb model can get 90% accuracy by guessing "Negative" every time. Always look at F1 instead.

### Loss
*   **What it is:** The "error penalty." The model tries to make this number zero.
*   **Trend:** It should **go down** over time.
    *   **Train Loss:** Should consistently decrease.
    *   **Val Loss:** Should decrease, then flatten out. If it starts going **up**, your model is "overfitting" (memorizing the test answers instead of learning).

### Precision
*   **Meaning:** "When it triggers, is it right?" (High Precision = Few False Alarms).

### Recall
*   **Meaning:** "Does it catch every attempt?" (High Recall = Few Missed Wakewords).

---

## 3. New Training Features (Google-Tier Upgrade)

The platform now includes advanced techniques to reach "Google-level" performance.

### 🧠 Knowledge Distillation (The Teacher)
*   **What it is:** We use a massive, smart brain (Wav2Vec 2.0) to teach your smaller model (MobileNet).
*   **Why use it?** It forces your small model to learn patterns it would miss on its own.
*   **How to enable:** Set `distillation.enabled = True` in config.

### 📉 Quantization Aware Training (QAT)
*   **What it is:** Training the model while pretending it's running on a cheap chip (INT8).
*   **Why use it?** If you plan to put this on an ESP32 or Arduino, this is mandatory.
*   **How to enable:** Set `qat.enabled = True` in config.

### 📏 Triplet Loss (Metric Learning)
*   **What it is:** A training method that pulls "wakeword" sounds closer together and pushes "confusing" sounds away.
*   **Why use it?** Reduces false alarms from similar words (e.g., "Hey Cat" vs "Hey Katya").
*   **How to enable:** Set `loss.loss_function = 'triplet_loss'`.

---

## 4. How to Read a Training Log

Here is an example from your log and what it means:

```text
Epoch 3 [Val]: Accuracy: 0.9065 | F1: 0.0270 | FPR: 0.0032 | FNR: 0.9859
```

*   **Accuracy (90%):** Looks high, but it's a lie!
*   **F1 (0.02):** Extremely low. This model is bad.
*   **FPR (0.3%):** Very low. It almost never triggers randomly (Good!).
*   **FNR (98%):** Extremely high. It misses 98% of your wakewords (Bad!).

**Diagnosis:** This model is too "shy." It is afraid to predict "Positive" because it doesn't want to be wrong. It needs more training or different parameters to become more confident.

---

## 5. Signs of a "Good" Training Run

1.  **Loss decreases steadily:** It doesn't jump around wildly.
2.  **F1 Score climbs:** It starts near 0 and grows to 0.8 or 0.9.
3.  **FPR stays low:** It doesn't explode to 10% or 20%.
4.  **FNR drops:** It starts high (missing everything) and drops below 10%.

## 6. Common Failure Patterns

| Symptom | Diagnosis | Solution |
| :--- | :--- | :--- |
| **F1 stays at 0.0** | Model is "dead." It predicts Negative for everything. | Learning rate is too high/low, or dataset is broken. |
| **Loss goes UP** | "Overfitting." Model is memorizing data. | Stop training early, increase Dropout, or get more data. |
| **FPR is huge (>20%)** | "Trigger Happy." Model thinks everything is a wakeword. | Add more background noise to your negative dataset. |
| **Loss is NaN** | "Exploding Gradients." The math broke. | Lower the learning rate significantly. |