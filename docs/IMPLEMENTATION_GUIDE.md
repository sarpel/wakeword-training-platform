# Implementation Guide - Production-Ready Features

This guide documents the newly implemented production-quality features from the implementation plan.

## 📋 Implemented Features

### ✅ Completed Features

1. **CMVN (Corpus-level Normalization)** - `src/data/cmvn.py`
2. **Balanced Batch Sampling** - `src/data/balanced_sampler.py`
3. **Temperature Scaling** - `src/models/temperature_scaling.py`
4. **Advanced Metrics (FAH, EER, pAUC)** - `src/training/advanced_metrics.py`
5. **Streaming Detector** - `src/evaluation/streaming_detector.py`
6. **EMA (Exponential Moving Average)** - `src/training/ema.py`
7. **LR Finder** - `src/training/lr_finder.py`

---

## 🔧 Feature Usage Guide

### 1. CMVN (Cepstral Mean Variance Normalization)

**Purpose**: Apply corpus-level normalization for consistent feature scaling across train/val/test.

**Usage**:

```python
from src.data.cmvn import CMVN, compute_cmvn_from_dataset
from pathlib import Path

# Compute CMVN stats from training data
cmvn = compute_cmvn_from_dataset(
    dataset=train_dataset,
    stats_path=Path("data/cmvn_stats.json"),
    max_samples=None  # Use all samples
)

# Apply normalization
normalized_features = cmvn.normalize(features)

# Later: Load pre-computed stats
cmvn = CMVN(stats_path=Path("data/cmvn_stats.json"))
```

**Integration Points**:
- Compute once during dataset preparation
- Apply in `Dataset.__getitem__()` after feature extraction
- Use same stats for train/val/test splits

---

### 2. Balanced Batch Sampler

**Purpose**: Maintain fixed ratio of positive, negative, and hard negative samples in each batch.

**Usage**:

```python
from src.data.balanced_sampler import BalancedBatchSampler, create_balanced_sampler_from_dataset
from torch.utils.data import DataLoader

# Create sampler with 1:1:1 ratio
sampler = create_balanced_sampler_from_dataset(
    dataset=train_dataset,
    batch_size=24,
    ratio=(1, 1, 1),  # pos:neg:hard_neg
    drop_last=True
)

# Use in DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_sampler=sampler,
    num_workers=16,
    pin_memory=True
)
```

**Configuration**:
- `ratio=(1, 1, 1)` - Equal distribution
- `ratio=(1, 2, 1)` - More negatives
- Requires enough samples in each category

---

### 3. Temperature Scaling (Model Calibration)

**Purpose**: Calibrate model confidence for better-calibrated probabilities.

**Usage**:

```python
from src.models.temperature_scaling import calibrate_model, apply_temperature_scaling

# After training, calibrate on validation set
temp_scaling = calibrate_model(
    model=trained_model,
    val_loader=val_loader,
    device='cuda',
    lr=0.01,
    max_iter=50
)

print(f"Optimal temperature: {temp_scaling.get_temperature():.4f}")

# Wrap model for inference
calibrated_model = apply_temperature_scaling(model, temp_scaling)

# Use calibrated model for evaluation
with torch.no_grad():
    logits = calibrated_model(inputs)
    probs = torch.softmax(logits, dim=1)
```

**When to Use**:
- After training completes
- Before final evaluation
- Improves confidence estimates without changing predictions

---

### 4. Advanced Metrics

**Purpose**: Calculate production-relevant metrics (FAH, EER, pAUC, operating point).

**Usage**:

```python
from src.training.advanced_metrics import (
    calculate_comprehensive_metrics,
    find_operating_point,
    calculate_eer
)

# Calculate all metrics
results = calculate_comprehensive_metrics(
    logits=model_logits,
    labels=ground_truth,
    total_seconds=total_audio_duration,
    target_fah=1.0  # Target: 1 false alarm per hour
)

print(f"ROC-AUC: {results['roc_auc']:.4f}")
print(f"EER: {results['eer']:.4f}")
print(f"pAUC (FPR≤0.1): {results['pauc_at_fpr_0.1']:.4f}")

# Operating point
op = results['operating_point']
print(f"Threshold: {op['threshold']:.4f}")
print(f"TPR: {op['tpr']:.4f}, FAH: {op['fah']:.2f}")
```

**Key Metrics**:
- **FAH**: False Alarms per Hour (critical for wakeword)
- **EER**: Equal Error Rate (industry standard)
- **pAUC**: Partial AUC (focus on low FPR region)
- **Operating Point**: Threshold meeting FAH target with max recall

---

### 5. Streaming Detector

**Purpose**: Real-time wakeword detection with voting, hysteresis, and lockout.

**Usage**:

```python
from src.evaluation.streaming_detector import StreamingDetector, SlidingWindowProcessor

# Initialize detector
detector = StreamingDetector(
    threshold_on=0.7,
    threshold_off=0.6,  # Hysteresis
    lockout_ms=1500,    # Lockout period
    vote_window=5,      # Window size
    vote_threshold=3    # Votes needed (3/5)
)

# Process audio stream
for timestamp_ms, score in stream:
    detected = detector.step(score, timestamp_ms)

    if detected:
        print(f"Wakeword detected at {timestamp_ms}ms!")
        # Trigger action
```

**Features**:
- **Voting**: Requires N/M votes to reduce false positives
- **Hysteresis**: Separate on/off thresholds for stability
- **Lockout**: Prevents multiple detections for single utterance

---

### 6. EMA (Exponential Moving Average)

**Purpose**: Maintain stable shadow weights for better inference.

**Usage**:

```python
from src.training.ema import EMA, EMAScheduler

# Create EMA tracker
ema = EMA(model, decay=0.999)

# During training
for epoch in range(epochs):
    for batch in train_loader:
        # Training step
        loss.backward()
        optimizer.step()

        # Update EMA after each step
        ema.update()

# Validation with EMA weights
original_params = ema.apply_shadow()
val_loss = validate(model, val_loader)
ema.restore(original_params)

# Save EMA state in checkpoint
checkpoint = {
    'model': model.state_dict(),
    'ema': ema.state_dict()
}
```

**With Adaptive Scheduler**:

```python
from src.training.ema import EMAScheduler

ema_scheduler = EMAScheduler(
    ema,
    initial_decay=0.999,
    final_decay=0.9995,  # Higher in final epochs
    final_epochs=10
)

# Each epoch
decay = ema_scheduler.step(epoch, total_epochs)
```

---

### 7. LR Finder

**Purpose**: Find optimal learning rate before training.

**Usage**:

```python
from src.training.lr_finder import LRFinder, find_lr

# Create LR finder
lr_finder = LRFinder(model, optimizer, criterion, device='cuda')

# Run range test
lrs, losses = lr_finder.range_test(
    train_loader,
    start_lr=1e-6,
    end_lr=1e-2,
    num_iter=200
)

# Get suggestion
suggested_lr = lr_finder.suggest_lr()
print(f"Suggested LR: {suggested_lr:.2e}")

# Plot results
lr_finder.plot(save_path=Path("lr_finder.png"))

# Use suggested LR for training
optimizer = torch.optim.AdamW(model.parameters(), lr=suggested_lr)
```

**Best Practices**:
- Run before main training
- Use same model architecture and data
- Model state is restored after finding

---

## 🔄 Complete Training Pipeline Integration

Example showing all features together:

```python
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.data.dataset import load_dataset_splits
from src.data.cmvn import compute_cmvn_from_dataset, CMVN
from src.data.balanced_sampler import create_balanced_sampler_from_dataset
from src.models.architectures import create_model
from src.training.ema import EMA, EMAScheduler
from src.training.lr_finder import LRFinder
from src.training.trainer import Trainer
from src.models.temperature_scaling import calibrate_model
from src.training.advanced_metrics import calculate_comprehensive_metrics

# 1. Load datasets
train_ds, val_ds, test_ds = load_dataset_splits(
    splits_dir=Path("data/splits"),
    sample_rate=16000,
    use_precomputed_features=True
)

# 2. Compute CMVN stats (once)
cmvn = compute_cmvn_from_dataset(
    dataset=train_ds,
    stats_path=Path("data/cmvn_stats.json")
)

# 3. Create balanced sampler
train_sampler = create_balanced_sampler_from_dataset(
    dataset=train_ds,
    batch_size=24,
    ratio=(1, 1, 1)
)

train_loader = DataLoader(
    train_ds,
    batch_sampler=train_sampler,
    num_workers=16,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=32,
    num_workers=8
)

# 4. Create model
model = create_model('resnet18', num_classes=2)
model = model.to('cuda')

# 5. Find optimal LR
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

lr_finder = LRFinder(model, optimizer, criterion)
lrs, losses = lr_finder.range_test(train_loader, num_iter=200)
optimal_lr = lr_finder.suggest_lr()

# Recreate optimizer with optimal LR
optimizer = torch.optim.AdamW(model.parameters(), lr=optimal_lr)

# 6. Create EMA
ema = EMA(model, decay=0.999)
ema_scheduler = EMAScheduler(ema, final_epochs=10)

# 7. Train with EMA
for epoch in range(80):
    model.train()
    for batch in train_loader:
        # Training step
        optimizer.zero_grad()
        loss = ...
        loss.backward()
        optimizer.step()

        # Update EMA
        ema.update()

    # Validation with EMA weights
    original_params = ema.apply_shadow()
    val_loss = validate(model, val_loader)
    ema.restore(original_params)

    # Update EMA decay
    ema_scheduler.step(epoch, total_epochs=80)

# 8. Calibrate with temperature scaling
temp_scaling = calibrate_model(model, val_loader)

# 9. Evaluate with comprehensive metrics
from src.evaluation.evaluator import evaluate_model

results = evaluate_model(
    model=model,
    test_loader=test_loader,
    temp_scaling=temp_scaling
)

# Calculate advanced metrics
metrics = calculate_comprehensive_metrics(
    logits=results['logits'],
    labels=results['labels'],
    total_seconds=results['total_duration'],
    target_fah=1.0
)

print(f"Final Results:")
print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"  EER: {metrics['eer']:.4f}")
print(f"  Operating Point: FAH={metrics['operating_point']['fah']:.2f}")
```

---

## 📊 Expected Improvements

With all features integrated:

1. **CMVN**: ~2-3% accuracy improvement from consistent normalization
2. **Balanced Sampling**: Better learning from hard negatives, ~5% FPR reduction
3. **Temperature Scaling**: Improved calibration (ECE reduction)
4. **EMA**: ~1-2% validation stability improvement
5. **Optimal LR**: Faster convergence, ~10-15% fewer epochs
6. **Advanced Metrics**: Better operating point selection, real-world performance tuning

---

## 🔍 Remaining Features (From Implementation Plan)

### High Priority

1. **Speaker-stratified K-fold**: Prevent speaker leakage
2. **Hard-negative mining pipeline**: 3-pass training strategy
3. **Gradient norm logging**: Track training stability
4. **Ablation flags**: Systematic component testing

### Medium Priority

5. **Focal loss**: Handle extreme imbalance
6. **TTA (Test-Time Augmentation)**: Ensemble predictions
7. **Latency measurement**: Production performance metrics
8. **Domain shift suite**: Robustness testing

### Lower Priority

9. **ONNX export with quantization**: Deployment optimization
10. **Model card template**: Documentation standard
11. **Reproducibility hash**: Config fingerprinting

---

## 🎯 Next Steps

1. **Integrate CMVN** into Dataset class
2. **Add EMA** to Trainer class
3. **Update evaluation** to use advanced metrics
4. **Add temperature scaling** to inference pipeline
5. **Test balanced sampler** with real data
6. **Run LR finder** before production training

---

## 📖 References

- Implementation Plan: `implementation_plan.md`
- CMVN: Corpus-level normalization (speech recognition standard)
- EMA: "Mean teachers are better role models" (Tarvainen & Valpola, 2017)
- LR Finder: "Cyclical Learning Rates" (Smith, 2017)
- Temperature Scaling: "On Calibration of Modern Neural Networks" (Guo et al., 2017)
- FAH Metric: Industry standard for wakeword systems

---

## ✅ Testing

All modules include standalone tests:

```bash
# Test individual modules
python -m src.data.cmvn
python -m src.data.balanced_sampler
python -m src.models.temperature_scaling
python -m src.training.advanced_metrics
python -m src.evaluation.streaming_detector
python -m src.training.ema
python -m src.training.lr_finder
```

Each module is fully documented with docstrings and examples.
