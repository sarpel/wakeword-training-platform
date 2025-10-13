# Quick Reference - New Features

Quick reference card for newly implemented production features.

---

## 🔍 Finding Modules

```
src/
├── data/
│   ├── cmvn.py                    # Corpus normalization
│   └── balanced_sampler.py        # Balanced batch sampling
├── models/
│   └── temperature_scaling.py     # Model calibration
├── training/
│   ├── advanced_metrics.py        # FAH, EER, pAUC metrics
│   ├── ema.py                     # Exponential moving average
│   └── lr_finder.py               # Learning rate finder
└── evaluation/
    └── streaming_detector.py      # Real-time detection
```

---

## ⚡ Quick Start Examples

### CMVN (Normalization)

```python
from src.data.cmvn import CMVN

# Compute once
cmvn = CMVN(stats_path="data/cmvn_stats.json")
cmvn.compute_stats(features_list, save=True)

# Apply everywhere
normalized = cmvn.normalize(features)
```

---

### Balanced Sampler

```python
from src.data.balanced_sampler import create_balanced_sampler_from_dataset

sampler = create_balanced_sampler_from_dataset(
    train_dataset, batch_size=24, ratio=(1,1,1)
)
loader = DataLoader(train_dataset, batch_sampler=sampler)
```

---

### Temperature Scaling

```python
from src.models.temperature_scaling import calibrate_model

# After training
temp_scaling = calibrate_model(model, val_loader)
print(f"T = {temp_scaling.get_temperature():.4f}")

# Inference
logits = temp_scaling(model(inputs))
```

---

### Advanced Metrics

```python
from src.training.advanced_metrics import calculate_comprehensive_metrics

metrics = calculate_comprehensive_metrics(
    logits, labels, total_seconds, target_fah=1.0
)

print(f"EER: {metrics['eer']:.4f}")
print(f"FAH: {metrics['operating_point']['fah']:.2f}")
```

---

### Streaming Detector

```python
from src.evaluation.streaming_detector import StreamingDetector

detector = StreamingDetector(
    threshold_on=0.7, lockout_ms=1500,
    vote_window=5, vote_threshold=3
)

for timestamp_ms, score in stream:
    if detector.step(score, timestamp_ms):
        print(f"Detection at {timestamp_ms}ms!")
```

---

### EMA

```python
from src.training.ema import EMA

ema = EMA(model, decay=0.999)

# Training
for batch in loader:
    optimizer.step()
    ema.update()

# Validation
original = ema.apply_shadow()
validate(model, val_loader)
ema.restore(original)
```

---

### LR Finder

```python
from src.training.lr_finder import find_lr

optimal_lr = find_lr(
    model, train_loader, optimizer, criterion,
    num_iter=200
)
print(f"Use LR: {optimal_lr:.2e}")
```

---

## 📊 Metrics Reference

| Metric | Range | Better | Description |
|--------|-------|--------|-------------|
| FAH | 0-∞ | Lower | False Alarms per Hour |
| EER | 0-1 | Lower | Equal Error Rate |
| pAUC | 0-1 | Higher | Partial AUC (FPR≤0.1) |
| ROC-AUC | 0-1 | Higher | Full ROC curve area |
| TPR | 0-1 | Higher | True Positive Rate |
| FPR | 0-1 | Lower | False Positive Rate |

---

## 🎯 Typical Operating Points

```python
# Consumer device (strict)
target_fah = 0.5  # 1 FA per 2 hours

# Smart speaker (balanced)
target_fah = 1.0  # 1 FA per hour

# Mobile app (lenient)
target_fah = 2.0  # 2 FA per hour
```

---

## ⚙️ Default Parameters

```python
# CMVN
eps = 1e-8

# Balanced Sampler
ratio = (1, 1, 1)  # pos:neg:hard_neg

# Temperature Scaling
lr = 0.01
max_iter = 50

# EMA
decay_initial = 0.999
decay_final = 0.9995

# LR Finder
start_lr = 1e-6
end_lr = 1e-2
num_iter = 200

# Streaming Detector
vote_window = 5
vote_threshold = 3
lockout_ms = 1500
```

---

## 🐛 Common Issues

### CMVN not reducing variance?
- Check that stats are computed on training set only
- Verify same stats used for all splits
- Ensure features extracted consistently

### Balanced Sampler has unequal batches?
- Need sufficient samples in each category
- Check `len(idx_pos)`, `len(idx_neg)`, `len(idx_hard_neg)`
- May need to reduce batch size

### Temperature > 2.0?
- Model severely uncalibrated
- Check validation set quality
- May need more training epochs

### LR Finder suggests very low LR?
- Loss may not be decreasing
- Check data pipeline and labels
- Try shorter num_iter

### Streaming Detector not triggering?
- Check threshold vs typical scores
- Reduce vote_threshold (e.g., 2/5 instead of 3/5)
- Verify score range [0,1]

---

## 📚 Full Documentation

- **Usage Guide**: `IMPLEMENTATION_GUIDE.md`
- **Summary**: `IMPLEMENTATION_SUMMARY.md`
- **Original Plan**: `implementation_plan.md`

---

## 🧪 Testing

```bash
# Test all modules
python -m src.data.cmvn
python -m src.data.balanced_sampler
python -m src.models.temperature_scaling
python -m src.training.advanced_metrics
python -m src.evaluation.streaming_detector
python -m src.training.ema
python -m src.training.lr_finder
```

---

## 💡 Best Practices

1. **Always** compute CMVN on training set only
2. **Always** use same temperature scaling for inference
3. **Always** run LR finder before production training
4. **Validate** EMA improves performance before using
5. **Test** streaming detector with realistic audio
6. **Monitor** FAH on realistic test scenarios
7. **Calibrate** threshold based on target FAH, not accuracy

---

## 🎓 Integration Order

1. LR Finder (pre-training)
2. CMVN (data pipeline)
3. Balanced Sampler (data loading)
4. EMA (training loop)
5. Temperature Scaling (post-training)
6. Advanced Metrics (evaluation)
7. Streaming Detector (deployment)

---

*Last Updated: 2025-10-12*
