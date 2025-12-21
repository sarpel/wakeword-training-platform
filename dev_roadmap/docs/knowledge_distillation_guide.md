# 📚 Knowledge Distillation Guide: Teacher-Student Training

## Table of Contents
1. [What is Knowledge Distillation? (ELI5)](#what-is-knowledge-distillation-eli5)
2. [How It Works in This Project](#how-it-works-in-this-project)
3. [Architecture Overview](#architecture-overview)
4. [Configuration Guide](#configuration-guide)
5. [Step-by-Step Usage](#step-by-step-usage)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Topics](#advanced-topics)

---

## What is Knowledge Distillation? (ELI5)

### 🧠 The Simple Explanation

Think of knowledge distillation like this:

**Teacher Model** = An experienced professor who knows a lot
**Student Model** = A young student learning from the professor

Instead of the student learning ONLY from textbooks (training data), the student ALSO learns by watching HOW the professor thinks about problems. This makes the student smarter, even though they have a smaller brain!

### 📊 Why Use It?

1. **Better Performance**: Student learns from both data AND teacher's wisdom
2. **Smaller Models**: Student can be much smaller but still perform well
3. **Faster Inference**: Small student model runs faster on devices (phones, IoT)
4. **Edge Deployment**: Perfect for deploying on resource-constrained devices

### 🔬 The Technical Explanation

Knowledge distillation transfers knowledge from a large, complex model (teacher) to a smaller, simpler model (student) by:

1. **Soft Targets**: Teacher outputs probability distributions (soft labels)
2. **Temperature Scaling**: Makes probabilities "softer" to reveal more information
3. **Combined Loss**: Student learns from both hard labels (ground truth) and soft labels (teacher)

**Formula:**
```
Total Loss = (1 - α) × Student_Loss + α × KL_Divergence(Student || Teacher)
```

Where:
- `α` (alpha) = Weight for distillation loss (0.0 to 1.0)
- `KL_Divergence` = Measure of difference between student and teacher predictions
- `Student_Loss` = Standard classification loss (cross-entropy)

---

## How It Works in This Project

### 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                         │
└─────────────────────────────────────────────────────────────┘

INPUT: Raw Audio Waveform (batch, samples)
  ↓
┌─────────────────────────────────────────────────────────────┐
│  TEACHER MODEL (Wav2Vec2)                                    │
│  - Large pretrained model (~95M parameters)                  │
│  - Processes raw audio directly                              │
│  - Outputs: Soft probability distribution                    │
│  - Status: Frozen (no gradient updates)                      │
└─────────────────────────────────────────────────────────────┘
  ↓ (teacher_logits)
  ↓
┌─────────────────────────────────────────────────────────────┐
│  STUDENT MODEL (MobileNetV3 / ResNet / etc.)                 │
│  - Small, efficient model (~1-20M parameters)                │
│  - Processes spectrograms OR raw audio                       │
│  - Outputs: Predicted probability distribution               │
│  - Status: Learning (gradients enabled)                      │
└─────────────────────────────────────────────────────────────┘
  ↓ (student_logits)
  ↓
┌─────────────────────────────────────────────────────────────┐
│  COMBINED LOSS CALCULATION                                   │
│                                                              │
│  1. Student Loss (standard cross-entropy)                    │
│     ├─ Compares student predictions vs ground truth         │
│     └─ Weight: (1 - alpha)                                   │
│                                                              │
│  2. Distillation Loss (KL divergence)                        │
│     ├─ Compares student predictions vs teacher predictions  │
│     ├─ Temperature scaling applied to soften distributions  │
│     └─ Weight: alpha                                         │
│                                                              │
│  Total = (1-α)×Student_Loss + α×Distillation_Loss           │
└─────────────────────────────────────────────────────────────┘
  ↓
BACKPROPAGATION (only updates student model)
```

### 🔑 Key Components

#### 1. **Teacher Model** (`src/models/huggingface.py`)
```python
class Wav2VecWakeword(nn.Module):
    # Uses HuggingFace's Wav2Vec2 as feature extractor
    # Pretrained on 960 hours of speech data
    # Expects RAW AUDIO input (batch, samples)
    # Output: logits (batch, num_classes)
```

**Important**: Teacher ONLY works with raw audio, NOT spectrograms!

#### 2. **Student Model** (Any architecture)
- MobileNetV3 (lightweight for edge)
- ResNet18 (balanced accuracy/speed)
- TinyConv (ultra-lightweight for MCU)
- LSTM/GRU/TCN (sequence models)

#### 3. **Distillation Trainer** (`src/training/distillation_trainer.py`)
- Extends base `Trainer` class
- Adds teacher model initialization
- Overrides `compute_loss()` to add distillation

---

## Configuration Guide

### 📋 Configuration Parameters

The distillation settings are in `DistillationConfig` (src/config/defaults.py:220-233):

```python
@dataclass
class DistillationConfig:
    """Knowledge Distillation configuration"""

    # Enable/disable distillation
    enabled: bool = False

    # Path to pretrained teacher checkpoint (optional)
    # Leave empty to use pretrained HuggingFace model
    teacher_model_path: str = ""

    # Teacher architecture (currently only "wav2vec2")
    teacher_architecture: str = "wav2vec2"

    # Temperature for softening probability distributions
    # Higher = softer (more information transfer)
    # Typical range: 1.0 - 10.0
    # Recommended: 2.0 - 4.0
    temperature: float = 2.0

    # Weight for distillation loss vs student loss
    # 0.0 = no distillation (only student loss)
    # 1.0 = only distillation (no student loss)
    # Recommended: 0.3 - 0.7
    alpha: float = 0.5
```

### 🎛️ Parameter Tuning Guide

| Parameter | Default | Typical Range | Effect |
|-----------|---------|---------------|--------|
| `enabled` | `False` | `True/False` | Enable distillation |
| `temperature` | `2.0` | `1.0 - 10.0` | Higher = softer targets, more knowledge transfer |
| `alpha` | `0.5` | `0.3 - 0.7` | Balance between teacher and ground truth |
| `teacher_model_path` | `""` | Any path | Use custom teacher weights |

**Recommendations:**
- **Small datasets**: Higher alpha (0.6-0.7) - rely more on teacher
- **Large datasets**: Lower alpha (0.3-0.4) - rely more on data
- **Strong teacher**: Higher alpha (0.6-0.8)
- **Weak teacher**: Lower alpha (0.2-0.4)

---

## Step-by-Step Usage

### Method 1: Using Configuration File (YAML)

#### Step 1: Create Configuration

Create a file `config/distillation_config.yaml`:

```yaml
# Distillation Configuration Example
config_name: "distillation_mobilenet"
description: "MobileNetV3 student learning from Wav2Vec2 teacher"

# Enable distillation
distillation:
  enabled: true
  teacher_architecture: "wav2vec2"
  teacher_model_path: ""  # Empty = use pretrained HuggingFace model
  temperature: 3.0        # Higher temp for more knowledge transfer
  alpha: 0.6              # 60% teacher, 40% ground truth

# Student model configuration
model:
  architecture: "mobilenetv3"  # Small, efficient student
  num_classes: 2
  pretrained: false
  dropout: 0.25

# Training configuration
training:
  batch_size: 32
  epochs: 80
  learning_rate: 0.001
  early_stopping_patience: 15

# Data configuration
data:
  sample_rate: 16000
  audio_duration: 1.5
  n_mels: 64
  feature_type: "mel"

  # CRITICAL: Must use raw audio for distillation!
  use_precomputed_features_for_training: false
  fallback_to_audio: true

# Other configs...
augmentation:
  background_noise_prob: 0.5
  rir_prob: 0.3

optimizer:
  optimizer: "adamw"
  weight_decay: 0.01
  scheduler: "cosine"
```

#### Step 2: Load and Train

```python
from src.config.defaults import WakewordConfig
from src.training.distillation_trainer import DistillationTrainer

# Load configuration
config = WakewordConfig.load("config/distillation_config.yaml")

# Verify distillation is enabled
print(f"Distillation enabled: {config.distillation.enabled}")
print(f"Teacher: {config.distillation.teacher_architecture}")
print(f"Alpha: {config.distillation.alpha}")

# Create trainer (automatically uses DistillationTrainer)
trainer = DistillationTrainer(
    model=student_model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    checkpoint_manager=checkpoint_manager,
    device="cuda"
)

# Train
trainer.train()
```

### Method 2: Using Python API Directly

```python
from src.config.defaults import WakewordConfig, DistillationConfig
from src.training.distillation_trainer import DistillationTrainer

# Create configuration
config = WakewordConfig()

# Configure distillation
config.distillation = DistillationConfig(
    enabled=True,
    teacher_architecture="wav2vec2",
    teacher_model_path="",  # Use pretrained
    temperature=3.0,
    alpha=0.6
)

# Configure student model
config.model.architecture = "mobilenetv3"
config.model.num_classes = 2
config.model.dropout = 0.25

# CRITICAL: Ensure raw audio usage
config.data.use_precomputed_features_for_training = False
config.data.fallback_to_audio = True

# Create and train
trainer = DistillationTrainer(
    model=student_model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    checkpoint_manager=checkpoint_manager,
    device="cuda"
)

results = trainer.train()
```

### Method 3: Using Gradio UI

#### Step 1: Launch UI
```bash
python -m src.ui.app
```

#### Step 2: Configure in Panel 2 (Configuration)

1. Go to "Optimization & Loss" tab
2. Find "Advanced Optimization" section
3. Enable distillation:
   - ✅ **Enable Distillation**
   - **Teacher Arch**: `wav2vec2`
   - **Distillation Temp**: `3.0`
   - **Distillation Alpha**: `0.6`

#### Step 3: Start Training in Panel 3

The trainer will automatically use `DistillationTrainer` when enabled.

---

## Examples

### Example 1: Training MobileNetV3 with Distillation

**Scenario**: Deploy lightweight wakeword model on mobile device

```python
# ============================================================
# EXAMPLE 1: MobileNetV3 Student Learning from Wav2Vec2 Teacher
# ============================================================

import torch
from pathlib import Path
from src.config.defaults import WakewordConfig, DistillationConfig
from src.training.distillation_trainer import DistillationTrainer
from src.models.factory import create_model
from src.data.dataset import WakewordDataset
from torch.utils.data import DataLoader

# Step 1: Configuration
# ---------------------
# This sets up all the hyperparameters for training
config = WakewordConfig()

# Enable knowledge distillation
# The teacher (Wav2Vec2) will guide the student (MobileNetV3)
config.distillation = DistillationConfig(
    enabled=True,                    # Turn on distillation
    teacher_architecture="wav2vec2",  # Use Wav2Vec2 as teacher
    temperature=3.0,                  # Higher temp = softer targets
    alpha=0.6                         # 60% teacher, 40% ground truth
)

# Student model: Small and efficient for mobile
config.model.architecture = "mobilenetv3"  # ~1.5M parameters
config.model.num_classes = 2               # Binary: wakeword vs non-wakeword
config.model.dropout = 0.25                # Prevent overfitting

# Data processing: MUST use raw audio for teacher
config.data.sample_rate = 16000
config.data.audio_duration = 1.5
config.data.n_mels = 64
config.data.use_precomputed_features_for_training = False  # CRITICAL!
config.data.fallback_to_audio = True                       # CRITICAL!

# Training parameters
config.training.batch_size = 32
config.training.epochs = 80
config.training.learning_rate = 0.001

# Step 2: Create Student Model
# -----------------------------
# Calculate input size based on audio parameters
time_steps = int(config.data.sample_rate * config.data.audio_duration) // config.data.hop_length + 1
input_size = config.data.n_mels  # For MobileNetV3

student_model = create_model(
    architecture="mobilenetv3",
    num_classes=2,
    input_size=input_size
)

# Step 3: Load Data with Raw Audio
# ---------------------------------
# Create datasets that return RAW AUDIO (not spectrograms)
# This is required for the teacher model
train_dataset = WakewordDataset(
    manifest_path=Path("data/splits/train.json"),
    sample_rate=16000,
    audio_duration=1.5,
    augment=True,
    return_raw_audio=True,  # CRITICAL: Teacher needs raw audio
    device="cuda"
)

val_dataset = WakewordDataset(
    manifest_path=Path("data/splits/val.json"),
    sample_rate=16000,
    audio_duration=1.5,
    augment=False,
    return_raw_audio=True,  # CRITICAL
    device="cuda"
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Step 4: Initialize Distillation Trainer
# ----------------------------------------
# The trainer will automatically load the teacher model
from src.training.checkpoint_manager import CheckpointManager

checkpoint_manager = CheckpointManager(Path("checkpoints"))

trainer = DistillationTrainer(
    model=student_model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    checkpoint_manager=checkpoint_manager,
    device="cuda"
)

# The teacher is automatically initialized here:
# - Loads pretrained Wav2Vec2 from HuggingFace
# - Freezes all teacher parameters
# - Moves teacher to GPU

# Step 5: Train
# -------------
print("Starting distillation training...")
print(f"Teacher: Wav2Vec2 (~95M params, frozen)")
print(f"Student: MobileNetV3 (~1.5M params, learning)")
print(f"Temperature: {config.distillation.temperature}")
print(f"Alpha: {config.distillation.alpha}")

results = trainer.train()

print(f"\nTraining complete!")
print(f"Best F1 Score: {results['best_f1']:.4f}")
print(f"Best Model: {results['best_checkpoint']}")

# Step 6: Export Student Model
# -----------------------------
# Export ONLY the student model (no teacher needed for inference)
student_model.eval()
dummy_input = torch.randn(1, 1, 64, time_steps).cuda()  # Spectrogram input

torch.onnx.export(
    student_model,
    dummy_input,
    "models/mobilenet_distilled.onnx",
    input_names=["audio_features"],
    output_names=["logits"],
    dynamic_axes={"audio_features": {0: "batch"}}
)

print("Student model exported to ONNX!")
```

### Example 2: Training TinyConv for ESP32 (Ultra-Lightweight)

**Scenario**: Deploy on ESP32-S3 microcontroller with strict memory limits

```python
# ============================================================
# EXAMPLE 2: TinyConv for ESP32 with Distillation
# Ultra-lightweight model (~60KB parameters)
# ============================================================

from src.config.presets import get_esp32_no_psram_preset

# Load ESP32 preset (already optimized)
config = get_esp32_no_psram_preset()

# Enable distillation to boost tiny model accuracy
config.distillation = DistillationConfig(
    enabled=True,
    teacher_architecture="wav2vec2",
    temperature=4.0,  # Higher temp for tiny student
    alpha=0.7         # Rely heavily on teacher (70%)
)

# Student model: TinyConv (~60K parameters)
config.model.architecture = "tiny_conv"
config.model.num_classes = 2
config.model.dropout = 0.2

# Data: Reduced features for memory
config.data.n_mels = 40          # Reduced from 64
config.data.audio_duration = 1.0  # Shorter audio

# Training: More epochs for tiny model
config.training.epochs = 150
config.training.batch_size = 32

# QAT enabled for INT8 deployment
config.qat.enabled = True
config.qat.backend = "qnnpack"  # ARM-friendly

# Create and train
trainer = DistillationTrainer(...)
trainer.train()

# Result:
# - TinyConv model with teacher's knowledge
# - ~60KB parameters (INT8 quantized)
# - Suitable for ESP32-S3 without PSRAM
```

### Example 3: Using Custom Teacher Checkpoint

**Scenario**: You've trained a powerful Wav2Vec2 model yourself

```python
# ============================================================
# EXAMPLE 3: Using Custom Teacher Checkpoint
# ============================================================

config = WakewordConfig()

# Point to your custom teacher checkpoint
config.distillation = DistillationConfig(
    enabled=True,
    teacher_architecture="wav2vec2",
    teacher_model_path="checkpoints/custom_teacher_best.pt",  # Your checkpoint
    temperature=2.5,
    alpha=0.5
)

# The trainer will load YOUR weights instead of HuggingFace pretrained
trainer = DistillationTrainer(...)

# Teacher loading process (automatic):
# 1. Initialize Wav2Vec2 architecture
# 2. Load weights from "checkpoints/custom_teacher_best.pt"
# 3. Freeze all parameters
# 4. Move to device
```

---

## Troubleshooting

### ❌ Common Issues and Solutions

#### Issue 1: "Teacher was NOT called with raw audio"
```
Error: Distillation skip verified: Teacher NOT called with spectrogram inputs.
```

**Cause**: Dataset returns spectrograms, but teacher needs raw audio.

**Solution**:
```python
# In your dataset configuration:
config.data.use_precomputed_features_for_training = False  # Disable precomputed
config.data.fallback_to_audio = True                       # Enable raw audio

# In WakewordDataset initialization:
dataset = WakewordDataset(
    return_raw_audio=True,  # ← ADD THIS!
    ...
)
```

#### Issue 2: "CUDA Out of Memory" with Distillation
```
RuntimeError: CUDA out of memory
```

**Cause**: Teacher model (Wav2Vec2) is large (~95M params) + student model.

**Solutions**:
1. **Reduce batch size**:
   ```python
   config.training.batch_size = 16  # Instead of 32
   ```

2. **Use gradient checkpointing** (advanced):
   ```python
   # In huggingface.py (modify Wav2Vec2 initialization)
   self.wav2vec2.gradient_checkpointing_enable()
   ```

3. **Mixed precision training** (already enabled by default):
   ```python
   config.optimizer.mixed_precision = True  # FP16 saves memory
   ```

#### Issue 3: Distillation Loss is NaN
```
Epoch 1: train_loss=nan, val_loss=nan
```

**Cause**: Temperature too high or numerical instability.

**Solutions**:
1. **Lower temperature**:
   ```python
   config.distillation.temperature = 2.0  # Instead of 10.0
   ```

2. **Check for zero division**:
   ```python
   # Ensure targets are valid
   assert not torch.isnan(teacher_logits).any()
   assert not torch.isnan(student_logits).any()
   ```

3. **Use gradient clipping**:
   ```python
   config.optimizer.gradient_clip = 1.0  # Prevent exploding gradients
   ```

#### Issue 4: Student Not Improving with Distillation
```
Validation accuracy stuck at ~50%
```

**Possible Causes & Solutions**:

1. **Alpha too high** (student ignores ground truth):
   ```python
   config.distillation.alpha = 0.3  # Lower alpha (30% teacher)
   ```

2. **Teacher not good enough**:
   ```python
   # Verify teacher accuracy first
   teacher_acc = evaluate_model(teacher, val_loader)
   print(f"Teacher accuracy: {teacher_acc:.2%}")
   # If < 80%, teacher may not be helpful
   ```

3. **Student capacity too small**:
   ```python
   # Try larger student model
   config.model.architecture = "resnet18"  # Instead of tiny_conv
   ```

#### Issue 5: "transformers library is required"
```
ImportError: transformers library is required for Wav2VecWakeword
```

**Solution**:
```bash
pip install transformers
```

#### Issue 6: Teacher and Student Batch Size Mismatch
```
RuntimeError: The size of tensor a (32) must match the size of tensor b (16)
```

**Cause**: Teacher and student process different batch sizes.

**Solution**: Ensure both see the same input:
```python
# In compute_loss (this should already be handled):
if inputs.dim() > 2:
    return student_loss  # Skip distillation for spectrograms
```

#### Issue 7: "compute_loss() got an unexpected keyword argument 'is_hard_negative'"
```
TypeError: DistillationTrainer.compute_loss() got an unexpected keyword argument 'is_hard_negative'
```

**Cause**: Older version of `DistillationTrainer` missing the `is_hard_negative` parameter.

**Solution**: Update to latest version or manually add parameter:
```python
# In distillation_trainer.py:
def compute_loss(
    self,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    inputs: Optional[torch.Tensor] = None,
    processed_inputs: Optional[torch.Tensor] = None,
    is_hard_negative: Optional[torch.Tensor] = None,  # ← ADD THIS
) -> torch.Tensor:
    # Pass is_hard_negative to parent
    student_loss = super().compute_loss(
        outputs, targets, inputs, processed_inputs, is_hard_negative
    )
    # ... rest of distillation logic
```

---

## Advanced Topics

### 🔬 Understanding the Loss Function

#### Mathematical Breakdown

```python
def compute_loss(self, outputs, targets, inputs):
    """
    Compute combined student + distillation loss

    Args:
        outputs: Student model predictions (batch, num_classes)
        targets: Ground truth labels (batch,)
        inputs: Raw audio waveform (batch, samples)

    Returns:
        total_loss: Combined loss value
    """

    # 1. STUDENT LOSS (Standard Cross-Entropy)
    # ========================================
    # This is the standard classification loss
    # Compares student predictions against ground truth
    student_loss = CrossEntropyLoss(outputs, targets)
    # Example: If student predicts [0.7, 0.3] but truth is [1, 0],
    #          loss will be high (prediction is wrong)

    # 2. TEACHER FORWARD PASS (No gradients)
    # =======================================
    with torch.no_grad():  # Don't update teacher
        teacher_logits = self.teacher(inputs)  # Raw audio → teacher

    # 3. TEMPERATURE SCALING
    # ======================
    # "Soften" the probability distributions
    T = self.config.distillation.temperature  # e.g., 2.0

    # Soft targets (teacher)
    # Example: [2.5, 0.1] / 2.0 = [1.25, 0.05] → [0.78, 0.22] (softer!)
    soft_targets = F.log_softmax(teacher_logits / T, dim=1)

    # Soft predictions (student)
    soft_student = F.log_softmax(outputs / T, dim=1)

    # 4. KL DIVERGENCE (How different are student and teacher?)
    # ==========================================================
    # Measures how different two probability distributions are
    # Lower KL = student matches teacher better
    distillation_loss = F.kl_div(
        soft_student,      # Student's soft predictions
        soft_targets,      # Teacher's soft targets
        reduction="batchmean",
        log_target=True
    ) * (T ** 2)  # Scale by T² to compensate for temperature

    # 5. COMBINE LOSSES
    # =================
    alpha = self.config.distillation.alpha  # e.g., 0.6

    total_loss = (1 - alpha) * student_loss + alpha * distillation_loss
    #            ↑                              ↑
    #            40% ground truth               60% teacher knowledge

    return total_loss
```

#### Why Temperature Scaling?

**Without temperature** (T=1.0):
- Teacher outputs: `[0.95, 0.05]` (very confident)
- Student learns: "Just predict class 0 always"
- **Problem**: No nuance, no generalization

**With temperature** (T=3.0):
- Teacher outputs: `[0.95, 0.05]` → Scaled: `[0.72, 0.28]`
- Student learns: "Prefer class 0, but class 1 has some signal"
- **Benefit**: Student learns subtle patterns and relationships

### 🎯 Choosing the Right Alpha Value

| Alpha | Teacher Weight | Ground Truth Weight | Best For |
|-------|---------------|---------------------|----------|
| 0.0 | 0% | 100% | No distillation (baseline) |
| 0.2 | 20% | 80% | Strong ground truth, weak teacher |
| 0.5 | 50% | 50% | Balanced (default) |
| 0.7 | 70% | 30% | Strong teacher, noisy labels |
| 1.0 | 100% | 0% | Pure imitation (risky!) |

**Experiment Strategy**:
```python
# Try multiple alpha values
for alpha in [0.3, 0.5, 0.7]:
    config.distillation.alpha = alpha
    train_and_evaluate(config)
    # Compare validation F1 scores
```

### 🧪 Ablation Study: Does Distillation Help?

Run controlled experiments:

```python
# Baseline (no distillation)
config_baseline = WakewordConfig()
config_baseline.distillation.enabled = False
results_baseline = train(config_baseline)

# With distillation
config_distill = WakewordConfig()
config_distill.distillation.enabled = True
config_distill.distillation.alpha = 0.6
results_distill = train(config_distill)

# Compare
print("Baseline F1:", results_baseline['f1'])
print("Distillation F1:", results_distill['f1'])
print("Improvement:", results_distill['f1'] - results_baseline['f1'])
```

### 🔍 Monitoring Distillation During Training

Add custom logging:

```python
class DistillationTrainer(Trainer):
    def compute_loss(self, outputs, targets, inputs):
        # ... (existing code) ...

        # Log individual loss components
        self.log_metrics({
            "student_loss": student_loss.item(),
            "distillation_loss": distillation_loss.item(),
            "total_loss": total_loss.item(),
            "loss_ratio": distillation_loss.item() / student_loss.item()
        })

        return total_loss
```

**What to watch**:
- `loss_ratio` should be stable (around 0.5-2.0)
- If ratio > 10: Student and teacher too different (reduce alpha)
- If ratio < 0.1: Teacher not contributing (increase alpha)

### 📊 Comparing Teacher and Student Predictions

```python
# Evaluate both models
teacher_results = evaluate_model(teacher, val_loader)
student_results = evaluate_model(student, val_loader)

print(f"Teacher - F1: {teacher_results['f1']:.3f}, Acc: {teacher_results['acc']:.3f}")
print(f"Student - F1: {student_results['f1']:.3f}, Acc: {student_results['acc']:.3f}")

# Goal: Student should be close to teacher (within 5% F1)
```

---

## Summary Checklist

Before starting distillation training, verify:

- [ ] **Configuration**
  - [ ] `distillation.enabled = True`
  - [ ] `temperature` in range 2.0-4.0
  - [ ] `alpha` in range 0.3-0.7

- [ ] **Data Pipeline**
  - [ ] `use_precomputed_features_for_training = False`
  - [ ] `fallback_to_audio = True`
  - [ ] `return_raw_audio = True` in dataset

- [ ] **Models**
  - [ ] Teacher model loaded correctly
  - [ ] Teacher parameters frozen
  - [ ] Student model smaller than teacher

- [ ] **Hardware**
  - [ ] GPU available (CPU too slow for Wav2Vec2)
  - [ ] Sufficient VRAM (8GB+ recommended)
  - [ ] Mixed precision enabled

- [ ] **Expected Behavior**
  - [ ] Training logs show both student and distillation loss
  - [ ] Loss values are not NaN
  - [ ] Validation accuracy improving

---

## References

1. **Original Paper**: Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
2. **HuggingFace Wav2Vec2**: https://huggingface.co/facebook/wav2vec2-base-960h
3. **Project Files**:
   - `src/training/distillation_trainer.py`: Main implementation
   - `src/models/huggingface.py`: Teacher model wrapper
   - `src/config/defaults.py`: Configuration dataclasses
   - `tests/test_distillation_trainer.py`: Unit tests

---

## Frequently Asked Questions

**Q: Can I use a different teacher model?**
A: Currently only Wav2Vec2 is supported. To add others, modify `_init_teacher()` in `distillation_trainer.py`.

**Q: Does distillation work with quantized models?**
A: Yes! Use QAT (Quantization-Aware Training) alongside distillation:
```python
config.qat.enabled = True
config.distillation.enabled = True
```

**Q: How much does distillation improve accuracy?**
A: Typical improvements: 2-5% absolute F1 score increase for small models.

**Q: Can I distill from multiple teachers?**
A: Not currently implemented, but theoretically possible (ensemble distillation).

**Q: What if my dataset is small (<1000 samples)?**
A: Distillation helps MORE with small data! Use higher alpha (0.7-0.8) to rely on teacher.

---

**Happy Distilling! 🎓→🎒**
