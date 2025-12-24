# FNR Optimization Changes - Summary

## Date: 2025-12-24

## Objective
Reduce False Negative Rate (FNR) below 2% for production deployment.

## Changes Applied

### 1. Configuration Changes (`src/config/defaults.py`)

#### TrainingConfig - Added FNR Target
```python
# Added field for FNR-oriented early stopping
fnr_target: Optional[float] = None  # Target FNR (e.g., 0.02 for 2%)
```

#### LossConfig - FNR-Oriented Parameters
```python
# CHANGED from cross_entropy to focal_loss
loss_function: str = "focal_loss"

# CHANGED: Focal alpha - FNR focused!
focal_alpha: float = 0.85  # Was: 0.25

# CHANGED: Focal gamma - Hard example mining
focal_gamma: float = 2.5  # Was: 2.0

# CHANGED: Hard negative weight
hard_negative_weight: float = 2.0  # Was: 1.5

# NEW: Dynamic alpha parameters
use_dynamic_alpha: bool = True
max_focal_alpha: float = 0.90  # Maximum alpha for dynamic scaling
```

### 2. Loss Function Changes (`src/models/losses.py`)

#### FocalLoss Class - Dynamic Alpha Support
```python
def set_alpha(self, alpha: float) -> None:
    """
    Update alpha parameter dynamically during training
    Used for FNR-oriented training with increasing alpha

    Args:
        alpha: New alpha value (0.0-1.0)
    """
    if self.alpha is not None:
        self.alpha = float(alpha)
        logger.debug(f"Focal alpha updated to {alpha:.3f}")
```

### 3. Trainer Changes (`src/training/trainer.py`)

#### Trainer.__init__ - Dynamic Alpha Initialization
```python
# Dynamic alpha for FNR optimization
self.use_dynamic_alpha = getattr(config.loss, "use_dynamic_alpha", False)
if self.use_dynamic_alpha:
    self.base_alpha = getattr(config.loss, "focal_alpha", 0.75)
    self.max_alpha = getattr(config.loss, "max_focal_alpha", 0.90)
    logger.info(f"Dynamic alpha enabled: {self.base_alpha} → {self.max_alpha}")
```

#### Trainer._compute_dynamic_alpha - New Method
```python
def _compute_dynamic_alpha(self, epoch: int) -> float:
    """
    Compute dynamic focal alpha based on training epoch

    Linear increase from base_alpha to max_alpha in first 50 epochs
    """
    if not self.use_dynamic_alpha:
        return getattr(self.config.loss, "focal_alpha", 0.25)

    if epoch < 50:
        progress = epoch / 50
        current_alpha = self.base_alpha + (self.max_alpha - self.base_alpha) * progress
    else:
        current_alpha = self.max_alpha

    return current_alpha
```

#### Trainer._check_improvement - FNR-Oriented Early Stopping
```python
# FNR-oriented early stopping if target is set
fnr_target = getattr(self.config.training, "fnr_target", None)
if fnr_target is not None:
    # FNR target mode - treat FNR as primary metric
    if val_fnr <= fnr_target:
        # FNR met, check F1 for best model
        if val_f1 > self.state.best_val_f1:
            self.state.best_val_f1 = val_f1
            improved = True
            self.state.epochs_without_improvement = 0
        else:
            self.state.epochs_without_improvement += 1
    else:
        # FNR still above target, keep training
        self.state.epochs_without_improvement = 0
        logger.debug(f"FNR={val_fnr:.4f} > target={fnr_target:.4f}, continuing training")
```

#### Trainer.train - Apply Dynamic Alpha
```python
# Apply dynamic alpha for FNR optimization
if self.use_dynamic_alpha and hasattr(self.criterion, "set_alpha"):
    current_alpha = self._compute_dynamic_alpha(epoch)
    self.criterion.set_alpha(current_alpha)
    logger.info(f"Epoch {epoch+1}: Dynamic focal alpha = {current_alpha:.4f}")
```

### 4. UI Changes (`src/ui/panel_training.py`)

#### Hard Negative Ratio - Default Value Changed
```python
# CHANGED: Default hard negative ratio from 1 to 2
sampler_ratio_hard = gr.Number(
    label="Hard Negative",
    value=2,  # Was: 1
    precision=0,
    minimum=0,
    info="Ratio of hard negatives (FNR optimization)",
)
```

## Impact Analysis

### Before Changes
- **Focal Alpha**: 0.25 (balanced)
- **Focal Gamma**: 2.0
- **Hard Negative Weight**: 1.5
- **Sampler Ratio**: 1:2:1 (Pos:Neg:HardNeg)
- **Loss Function**: cross_entropy

### After Changes
- **Focal Alpha**: 0.85 → 0.90 (dynamic)
- **Focal Gamma**: 2.5
- **Hard Negative Weight**: 2.0
- **Sampler Ratio**: 1:2:2 (Pos:Neg:HardNeg)
- **Loss Function**: focal_loss (with dynamic alpha)

## Expected Results

| Metric | Before | Target (After Changes) |
|--------|--------|----------------------|
| **FNR** | 2.38% | **< 2.0%** |
| **F1** | 0.7856 | **0.80+** |
| **EER** | 2.49% | **< 2.0%** |
| **FAH** | 110.18 | 120-150 (acceptable) |

## Training Recommendations

### 1. Clear Old Checkpoints
```bash
rm -rf models/checkpoints/*
rm models/best_model.pt
```

### 2. Start Fresh Training
Use the updated defaults or create a config with FNR target:
```yaml
training:
  fnr_target: 0.02  # 2% FNR target

loss:
  use_dynamic_alpha: true
  max_focal_alpha: 0.90
  hard_negative_weight: 2.0
```

### 3. Monitor Training
- Watch the first 50 epochs
- FNR should steadily decrease
- Look for "Dynamic focal alpha = X.XXXX" in logs
- Target: FNR < 2% by epoch 50

### 4. Verify After Training
- Quantize model with QAT (if enabled)
- Test FNR on validation set
- Adjust threshold if needed to meet FNR < 2%

## Technical Notes

### Why These Changes?

1. **Focal Alpha = 0.85**: Higher alpha gives more weight to the positive class, reducing false negatives
2. **Dynamic Alpha**: Starts moderate, increases to focus more on positives as training progresses
3. **Focal Gamma = 2.5**: Higher gamma focuses more on hard examples (false negatives)
4. **Hard Negative Weight = 2.0**: Gives hard negatives more importance in loss calculation
5. **Sampler Ratio 1:2:2**: Doubles hard negative exposure for better learning

### Arch Safeguards

- No breaking changes to existing models
- Backward compatible (old configs still work)
- Dynamic alpha is optional (can be disabled)
- FNR early stopping only activates when target is set

## Rollback Plan

If any issues occur, revert to:
```python
loss_function: str = "cross_entropy"
focal_alpha: float = 0.25
focal_gamma: float = 2.0
hard_negative_weight: float = 1.5
sampler_ratio_hard = 1
use_dynamic_alpha = False
```

## References
- Original requirements from training-153epoch-f1-0.83.txt log
- FNR target: < 2.38% → < 2.0%
- Best F1 achieved: 0.83

## Next Steps

1. ✅ Code changes completed
2. ⏳ Run training with new defaults
3. ⏳ Verify FNR < 2% on validation set
4. ⏳ Export quantized model if passes
5. ⏳ Deploy to production
