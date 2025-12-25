# Expert Guide: Advanced Knowledge Distillation & Feature Alignment

This guide provides deep technical insights into selecting teacher layers and configuring projectors for optimal knowledge transfer from large foundation models (Wav2Vec2, Conformer) to ultra-lightweight students (TinyConv).

## 1. The Power of Feature Alignment

Standard distillation only uses the **Logits** (final predictions) of the teacher. **Feature Alignment** goes deeper, forcing the student's internal "thought process" to match the teacher's intermediate representations.

### Benefits:
*   **Faster Convergence:** The student learns "what to look for" much earlier.
*   **Higher Robustness:** Learned features are more generalizable across different acoustic environments.
*   **Hierarchical Learning:** Different layers capture different levels of abstraction (e.g., phonemes vs. semantic patterns).

---

## 2. Layer Selection Strategy

In this platform, you can manually select which internal layers of the Teacher models to align with the Student.

### Wav2Vec 2.0 (Teacher 1)
Wav2Vec2 consists of a CNN feature encoder followed by 12 Transformer blocks.

| Layer Index | Level of Abstraction | Recommendation |
| :--- | :--- | :--- |
| **0** | CNN Latents | Good for capturing basic spectral patterns. |
| **1 - 4** | Lower Transformer | Focuses on phonetics and speech primitives. Highly recommended for Wakewords. |
| **5 - 8** | Mid Transformer | Captures temporal dependencies and syllable-like structures. |
| **9 - 12** | Upper Transformer | More semantic and task-specific. Less useful for simple triggers. |

**Pro Tip:** For short wakewords (e.g., "Hey Sirius"), focus on layers **2, 4, and 6**.

### Conformer (Teacher 2)
The Conformer combines local (CNN) and global (Transformer) insights.

| Layer Index | Strategic Value |
| :--- | :--- |
| **Early Layers** | Local acoustic features. Crucial for student models with small receptive fields (TinyConv). |
| **Middle Layers** | Balanced representation. Often the most stable for alignment. |
| **Final Layers** | High-level detection logic. Best aligned with the student's penultimate layer. |

---

## 3. How Learnable Projectors Work

When you align a **Student (dim 64)** with a **Teacher (dim 768)**, the system automatically injects a **Learnable Projector**.

1.  **Architecture:** A small MLP (Linear -> LayerNorm -> ReLU -> Linear).
2.  **Function:** It "translates" the student's 64 features into the teacher's 768-dimensional space.
3.  **Loss:** The system minimizes the **Mean Squared Error (MSE)** between the projected student features and the actual teacher features.
4.  **Training:** The projector is trained *simultaneously* with the student model but is discarded during inference (Zero latency impact on deployment).

---

## 4. Configuration Reference

In `panel_config.py` or your configuration YAML:

```yaml
distillation:
  enabled: true
  feature_alignment_enabled: true
  feature_alignment_weight: 0.1  # Start low, increase if student loss is stable
  alignment_layers: [2, 4, 6]    # Indices of teacher layers to align
```

### When to adjust `feature_alignment_weight`:
*   **Too High (> 0.5):** The student might struggle to satisfy its own classification loss as it tries too hard to copy the teacher.
*   **Too Low (< 0.01):** Feature alignment will have negligible impact.
*   **Sweet Spot:** Usually **0.05 to 0.2**.

---

## 5. Troubleshooting Alignment

*   **Diverging Loss:** If the loss becomes NaN or grows rapidly, reduce the `feature_alignment_weight` or the `learning_rate`.
*   **No Accuracy Gain:** Try different layers. If you aligned layer 12 (high-level), try layer 4 (low-level) instead. Lightweight students often benefit more from low-to-mid level teacher features.
*   **Memory Issues:** Multi-layer alignment increases VRAM usage during training. If you hit OOM, reduce the number of `alignment_layers`.
