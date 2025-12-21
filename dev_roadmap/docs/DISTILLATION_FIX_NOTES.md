# 🔧 Distillation Trainer Fix - December 18, 2025

## Issue Fixed

**Error**: `TypeError: DistillationTrainer.compute_loss() got an unexpected keyword argument 'is_hard_negative'`

## Root Cause

The `DistillationTrainer.compute_loss()` method signature was missing the `is_hard_negative` parameter that exists in the base `Trainer.compute_loss()` method.

The training loop (`training_loop.py`) passes this parameter to support hard negative mining (a technique to emphasize difficult negative examples during training), but the distillation trainer wasn't accepting it.

## What Changed

### Before (Broken)
```python
# src/training/distillation_trainer.py
def compute_loss(
    self,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    inputs: Optional[torch.Tensor] = None,
    processed_inputs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    student_loss = super().compute_loss(outputs, targets, inputs, processed_inputs)
    # ... distillation logic
```

### After (Fixed)
```python
# src/training/distillation_trainer.py
def compute_loss(
    self,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    inputs: Optional[torch.Tensor] = None,
    processed_inputs: Optional[torch.Tensor] = None,
    is_hard_negative: Optional[torch.Tensor] = None,  # ← ADDED
) -> torch.Tensor:
    # Pass is_hard_negative to parent for proper hard negative weighting
    student_loss = super().compute_loss(
        outputs, targets, inputs, processed_inputs, is_hard_negative
    )
    # ... distillation logic
```

## Files Modified

1. ✅ `src/training/distillation_trainer.py` - Updated method signature
2. ✅ `tests/test_distillation_trainer.py` - Updated test calls
3. ✅ `docs/knowledge_distillation_guide.md` - Added troubleshooting entry
4. ✅ `docs/distillation_quick_reference.md` - Added to troubleshooting table

## How This Affects You

### If You're Using Distillation

**Good news**: This fix is backward compatible. Your existing code will work without changes.

The `is_hard_negative` parameter is **optional** (defaults to `None`), so:
- Training without hard negatives: Works ✅
- Training with hard negatives: Now works ✅

### Hard Negative Mining

Hard negative mining improves model performance by giving more weight to difficult negative examples (e.g., similar-sounding non-wakewords).

**Example**:
```python
# When hard negative mining is enabled:
# - Normal negatives get weight 1.0
# - Hard negatives get weight 1.5 (configurable)
config.loss.hard_negative_weight = 1.5

# The is_hard_negative tensor marks which samples are hard
# is_hard_negative = [0, 1, 0, 0, 1]  # batch of 5
#                    ↑  ↑        ↑
#                    │  │        └─ Hard negative
#                    │  └─ Hard negative
#                    └─ Normal sample
```

With distillation, the weighted student loss is combined with the distillation loss:
```
Total Loss = (1-α) × Weighted_Student_Loss + α × KL_Divergence
                        ↑
                        Includes hard negative weighting
```

## Test Verification

All tests pass:
```bash
pytest tests/test_distillation_trainer.py -v
# ✅ test_distillation_logic PASSED
# ✅ test_distillation_skip_on_spectrograms PASSED
```

## Impact on Training

This fix **improves** distillation training quality when hard negative mining is enabled because:

1. **Before fix**: Hard negative weighting was ignored (parameter rejected)
2. **After fix**: Hard negative weighting properly applied to student loss

Result: Better handling of difficult negative examples during distillation training.

## Upgrade Instructions

### If You Modified distillation_trainer.py Yourself

Compare your version with the fixed version:
```bash
# Check if you have the fix
grep "is_hard_negative" src/training/distillation_trainer.py
```

If the parameter is missing, apply the fix shown above.

### If You're Using the Default Code

Simply pull the latest changes - the fix is already applied.

## Technical Details

### Method Signature Compatibility

Python method overriding requires matching signatures. The training loop calls:
```python
loss = trainer.compute_loss(
    outputs=student_outputs,
    targets=batch_targets,
    inputs=raw_audio,
    processed_inputs=features,
    is_hard_negative=hard_negative_mask  # ← This parameter MUST be accepted
)
```

If `DistillationTrainer.compute_loss()` doesn't accept `is_hard_negative`, Python raises `TypeError`.

### Why This Matters for Distillation

Hard negative mining is especially valuable when training with distillation because:

1. **Teacher knows** which examples are hard (from its own training)
2. **Student benefits** from emphasis on difficult cases
3. **Distillation loss** guides student on both easy and hard examples
4. **Combined effect**: Student learns teacher's wisdom + focuses on hard cases

## Related Configuration

```yaml
# Enable hard negative mining
loss:
  hard_negative_weight: 1.5  # Weight for hard negatives (1.0 = no weighting)

# Works seamlessly with distillation
distillation:
  enabled: true
  alpha: 0.6
  temperature: 3.0
```

## Questions?

See the full troubleshooting guide: `docs/knowledge_distillation_guide.md` (Issue #7)

---

**Status**: ✅ Fixed and verified
**Date**: December 18, 2025
**Tests**: All passing
