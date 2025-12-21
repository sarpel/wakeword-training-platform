# HPO Best Parameters - Profile Guide

## Summary

Successfully saved your HPO best parameters (F1 Score: **0.9712**) as reusable profiles!

### Files Created

All profiles are saved in `configs/profiles/`:

1. **hpo_best_training.json** - Training & Optimizer parameters
2. **hpo_best_model.json** - Model architecture parameters
3. **hpo_best_augmentation.json** - Data augmentation parameters
4. **hpo_best_loss.json** - Loss function configuration
5. **hpo_best_complete.json** - All parameters combined
6. **hpo_best_summary.json** - Metadata and usage guide

---

## Best HPO Parameters

From Optuna study with 50 trials:

### Training Parameters
- **Batch Size**: 32
- **Learning Rate**: 0.00514 (5.14e-3)
- **Optimizer**: AdamW
- **Weight Decay**: 2.94e-6

### Model Parameters
- **Dropout**: 0.167 (16.7%)

### Augmentation Parameters
- **Background Noise Probability**: 0.899 (89.9%)
- **RIR Probability**: 0.437 (43.7%)
- **Time Stretch Range**: 0.938 - 1.173
- **Frequency Mask Parameter**: 12
- **Time Mask Parameter**: 26

### Loss Function
- **Loss Function**: Cross Entropy

---

## How to Use These Profiles

### Option 1: Load in Gradio UI (When Available)

If your UI has profile loading functionality:
1. Navigate to the respective panel (Training, Model, Augmentation, etc.)
2. Click "Load Profile" button
3. Select the corresponding JSON file from `configs/profiles/`

### Option 2: Programmatic Loading

```python
import json
from pathlib import Path
from src.config.defaults import WakewordConfig

# Load complete profile
with open("configs/profiles/hpo_best_complete.json", 'r') as f:
    profile = json.load(f)

# Apply to config
config = WakewordConfig()

# Apply training parameters
config.training.batch_size = profile["parameters"]["training"]["batch_size"]
config.training.learning_rate = profile["parameters"]["training"]["learning_rate"]

# Apply optimizer parameters
config.optimizer.optimizer = profile["parameters"]["optimizer"]["optimizer"]
config.optimizer.weight_decay = profile["parameters"]["optimizer"]["weight_decay"]

# Apply model parameters
config.model.dropout = profile["parameters"]["model"]["dropout"]

# Apply augmentation parameters
aug_params = profile["parameters"]["augmentation"]
config.augmentation.background_noise_prob = aug_params["background_noise_prob"]
config.augmentation.rir_prob = aug_params["rir_prob"]
config.augmentation.time_stretch_min = aug_params["time_stretch_min"]
config.augmentation.time_stretch_max = aug_params["time_stretch_max"]
config.augmentation.freq_mask_param = aug_params["freq_mask_param"]
config.augmentation.time_mask_param = aug_params["time_mask_param"]

# Apply loss function
config.loss.loss_function = profile["parameters"]["loss"]["loss_function"]
```

### Option 3: Load Individual Parameter Groups

```python
import json

# Load only augmentation parameters
with open("configs/profiles/hpo_best_augmentation.json", 'r') as f:
    aug_profile = json.load(f)

# Apply to config
for key, value in aug_profile["parameters"]["augmentation"].items():
    setattr(config.augmentation, key, value)
```

---

## Bug Fix Applied

### Fixed KeyError in `apply_best_params`

**File**: `src/ui/panel_training.py` (line 1094)

**Issue**: Code was trying to access `best_params["rir_params"]` but the key was actually `"rir_prob"`

**Fix Applied**:
```python
# Before (WRONG)
config.augmentation.rir_prob = best_params["rir_params"]

# After (CORRECT)
config.augmentation.rir_prob = best_params["rir_prob"]
```

Now you can click "Apply Best Params" in the UI without errors!

---

## Performance Impact

Using these HPO-optimized parameters achieved:

- **F1 Score**: 0.9712 (97.12%)
- **Optimization**: 50 Optuna trials
- **Key Findings**:
  - Lower learning rate (0.00514 vs typical 0.001-0.0003)
  - Smaller batch size (32 vs typical 64)
  - Very low weight decay (2.94e-6)
  - High dropout (16.7%)
  - Aggressive augmentation (89.9% background noise)

---

## Next Steps

1. **Test the profiles**: Load and verify they work correctly
2. **Start training**: Use these parameters for your next training run
3. **Compare results**: Document performance vs. default parameters
4. **Fine-tune**: Use as baseline for further optimization

---

## Files Modified

1. `src/ui/panel_training.py` - Fixed KeyError bug
2. `src/config/paths.py` - Added PROFILES path
3. `scripts/save_hpo_profiles.py` - New profile saving script

## Files Created

All files in `configs/profiles/`:
- `hpo_best_training.json`
- `hpo_best_model.json`
- `hpo_best_augmentation.json`
- `hpo_best_loss.json`
- `hpo_best_complete.json`
- `hpo_best_summary.json`

---

**Generated**: 2025-12-19
**HPO Run**: 50 trials with Optuna
**Best Trial F1**: 0.971195391262602
