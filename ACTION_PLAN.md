# Comprehensive Action Plan - Wakeword Training Platform

This document consolidates ALL comments, suggestions, and action items from across the repository, organized by priority with expanded subcomments and detailed tasks.

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Priority Matrix](#priority-matrix)
3. [P0 - Critical (Blocking Issues)](#p0---critical-blocking-issues)
4. [P1 - High Priority (Security & Stability)](#p1---high-priority-security--stability)
5. [P2 - Medium Priority (Performance & Quality)](#p2---medium-priority-performance--quality)
6. [P3 - Low Priority (Code Quality & Maintenance)](#p3---low-priority-code-quality--maintenance)
7. [Active Development Tracks](#active-development-tracks)
8. [Testing Checklist](#testing-checklist)
9. [Timeline Estimate](#timeline-estimate)
10. [Quick Fix Scripts](#quick-fix-scripts)

---

## Executive Summary

| Category | Count | Status |
|----------|-------|--------|
| Critical Bugs (P0) | 47+ | 🔴 Blocking |
| Security Vulnerabilities (P1) | 8 | 🟠 High Risk |
| Performance Improvements (P2) | 15+ | 🟡 Important |
| Code Quality (P3) | 55+ | 🟢 Nice to Have |
| Active Tracks | 4 | 📝 In Progress |
| Archived Tracks | 12 | ✅ Completed/Deferred |

**Overall Assessment:** A- (85/100) - Production-ready with critical issues to address.

---

## Priority Matrix

| Priority | Estimated Time | Impact | Actions Required |
|----------|----------------|--------|------------------|
| **P0** | 1-2 days | Runtime errors fixed | 47 undefined names, startup fixes |
| **P1** | 1 day | Security hardened | 8 torch.load fixes, hash security |
| **P2** | 2-3 days | 2-4x performance gain | GPU optimization, caching |
| **P3** | 1-2 weeks | Clean codebase | Import cleanup, docs |

---

## P0 - Critical (Blocking Issues)

### 🔴 1. Undefined Name Errors (47 total)

These errors will cause `NameError` at runtime and must be fixed immediately.

#### 1.1 `src/export/onnx_exporter.py` (16 errors)
**Issue:** Lazy import pattern used but global scope references exist.

| Line | Variable | Fix |
|------|----------|-----|
| 47, 331, 343, 344, 488, 492 | `onnx` | Add `import onnx` at module level or inside functions |
| 47, 331, 363, 424, 455, 488, 493, 497 | `ort` | Add `import onnxruntime as ort` |
| 384, 385 | `np` | Add `import numpy as np` |

**Action:**
```python
# Add at top of file:
import numpy as np
import onnx
import onnxruntime as ort
```

#### 1.2 `src/evaluation/evaluator.py` (11 errors)
**Issue:** Missing imports and undefined functions.

| Line | Missing | Fix |
|------|---------|-----|
| 66 | `enforce_cuda` | `from src.config.cuda_utils import enforce_cuda` |
| 78 | `AudioProcessor` | `from src.data.audio_utils import AudioProcessor` |
| 88 | `FeatureExtractor` | `from src.data.feature_extraction import FeatureExtractor` |
| 99 | `MetricsCalculator` | `from src.training.metrics import MetricsCalculator` |
| 104 | `evaluate_file` | Implement or import function |
| 107 | `evaluate_files` | Implement or import function |
| 110 | `evaluate_dataset` | Implement or import function |
| 113 | `get_roc_curve_data` | Implement or import function |
| 116 | `evaluate_with_advanced_metrics` | Implement or import function |

**Action:**
```python
# Add imports at top of file:
import time
from src.config.cuda_utils import enforce_cuda
from src.data.audio_utils import AudioProcessor
from src.data.feature_extraction import FeatureExtractor
from src.training.metrics import MetricsCalculator
```

#### 1.3 `src/ui/panel_export.py` (5 errors)
| Line | Missing | Fix |
|------|---------|-----|
| 102, 171 | `time` | `import time` |
| 112 | `export_model_to_onnx` | Import from onnx_exporter |
| 230 | `validate_onnx_model` | Import from onnx_exporter |
| 260 | `benchmark_onnx_model` | Import from onnx_exporter |

#### 1.4 `src/ui/panel_evaluation.py` (5 errors)
| Line | Missing | Fix |
|------|---------|-----|
| 273, 404 | `time` | `import time` |
| 332 | `SimulatedMicrophoneInference` | Import from inference module |
| 475 | `WakewordDataset` | `from src.data.dataset import WakewordDataset` |
| 571 | `MetricResults` | Import from metrics module |

#### 1.5 `src/evaluation/dataset_evaluator.py` (3 errors)
| Line | Missing | Fix |
|------|---------|-----|
| 63, 70 | `time` | `import time` |
| 86 | `Path` | `from pathlib import Path` |

#### 1.6 `src/training/checkpoint_manager.py` (3 errors)
| Line | Missing | Fix |
|------|---------|-----|
| 57 | `Trainer` | Use `TYPE_CHECKING` for type hints |
| 328 | `json` | `import json` |
| 551 | `shutil` | `import shutil` |

**Action:**
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.training.trainer import Trainer
import json
import shutil
```

#### 1.7 `src/training/checkpoint.py` (3 errors)
| Line | Missing | Fix |
|------|---------|-----|
| 8, 55 | `Trainer` | Use `TYPE_CHECKING` |
| 11 | `MetricResults` | Import from metrics |

#### 1.8 `src/evaluation/advanced_evaluator.py` (1 error)
| Line | Missing | Fix |
|------|---------|-----|
| 68 | `calculate_comprehensive_metrics` | Implement or import function |

#### 1.9 `src/config/logger.py` (1 error)
| Line | Issue | Fix |
|------|-------|-----|
| 45 | `get_logger` undefined in `__main__` | Change to `get_data_logger` |

#### 1.10 `src/data/dataset.py` (1 error)
| Line | Issue | Fix |
|------|-------|-----|
| 549 | `splits_dir` out of scope in `__main__` | Change to `data_root / 'splits'` |

---

### 🔴 2. Startup Failures (Critical Bug)

**Source:** `conductor/tracks/startup_hpo_fixes_20251223/spec.md`

#### 2.1 Application Launch Failure
**Issue:** `NameError: name 'results' is not defined` after Evaluation Panel initialization.

**Location:** `run.py` or `src/ui/app.py`

**Action:**
- [ ] Locate `results` reference in startup sequence
- [ ] Ensure variables are properly defined/scoped
- [ ] Verify application reaches Gradio UI

#### 2.2 HPO Trial Crash
**Issue:** `UnboundLocalError` for `best_f1` and `NotImplementedError` for `Trial.report`.

**Location:** `src/training/hpo.py`

**Actions:**
- [ ] Initialize `best_f1` before checkpoint check logic
- [ ] Fix Optuna compatibility for single-objective optimization
- [ ] Remove `trial.report` calls for multi-objective studies

---

## P1 - High Priority (Security & Stability)

### 🟠 1. Security Vulnerabilities

#### 1.1 Unsafe PyTorch Model Loading (CWE-502)
**Risk:** Pickle deserialization attack allowing arbitrary code execution.

| File | Line | Current Code |
|------|------|--------------|
| `src/evaluation/evaluator.py` | 138 | `torch.load(path, map_location=device)` |
| `src/export/onnx_exporter.py` | 238 | `torch.load(path, map_location=device)` |
| `src/training/checkpoint.py` | 59 | `torch.load(path, map_location=device)` |
| `src/training/checkpoint_manager.py` | 131, 216, 380 | `torch.load(path, map_location=device)` |

**Fix:**
```python
# Replace ALL instances with (PyTorch 1.13+):
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

# For compatibility with older PyTorch versions:
import torch
from packaging import version

def safe_torch_load(path, map_location=None):
    """Load checkpoint with weights_only=True if supported (PyTorch >= 1.13)."""
    if version.parse(torch.__version__) >= version.parse("1.13.0"):
        return torch.load(path, map_location=map_location, weights_only=True)
    else:
        return torch.load(path, map_location=map_location)

checkpoint = safe_torch_load(checkpoint_path, map_location=device)
```

#### 1.2 Weak Hash Algorithm (CWE-327)
**File:** `src/data/file_cache.py` - Line 73

**Current:**
```python
key_hash = hashlib.md5(key_data.encode()).hexdigest()
```

**Fix:**
```python
# Option 1: Use stronger hash
key_hash = hashlib.sha256(key_data.encode()).hexdigest()

# Option 2: Mark as non-security use
key_hash = hashlib.md5(key_data.encode(), usedforsecurity=False).hexdigest()
```

#### 1.3 Path Traversal Vulnerability (CVSS 7.5)
**File:** `src/ui/panel_dataset.py` - Line 48

**Issue:** User-controlled path allows directory traversal.

**Fix:**
```python
def sanitize_path(user_path: str, allowed_root: Path) -> Path:
    """Validate path is within allowed directory"""
    full_path = Path(user_path).resolve()
    allowed_root = allowed_root.resolve()
    
    if not str(full_path).startswith(str(allowed_root)):
        raise ValueError(f"Path {user_path} outside allowed directory")
    
    return full_path
```

#### 1.4 Arbitrary Code Execution via Pickle
**File:** `src/data/dataset.py` - Line 221

**Fix:**
```python
# Add allow_pickle=False for untrusted data
features = np.load(npy_path, mmap_mode='r', allow_pickle=False)
```

#### 1.5 Unvalidated File Deletion
**File:** `src/ui/panel_dataset.py` - Line 169-172

**Issue:** No confirmation dialog for "Delete Invalid Files" checkbox.

**Fix:**
- Add confirmation dialog
- Implement backup mechanism before deletion

#### 1.6 Information Disclosure
**File:** `src/config/validator.py` - Line 326-331

**Fix:**
```python
# Replace print() with proper logging
logger.warning("Self-test skipped", exc_info=True)
```

---

### 🟠 2. Broad Exception Handling (71 locations)

**Issue:** `except Exception:` catches all exceptions including `KeyboardInterrupt`, `SystemExit`.

**Files Affected:**
- `src/data/file_cache.py` (4 instances)
- `src/data/batch_feature_extractor.py` (3 instances)
- `src/training/trainer.py` (5 instances)
- `src/ui/panel_*.py` (20+ instances)
- `src/config/validator.py` (3 instances: lines 140, 169, 441)
- `src/data/splitter.py` (2 instances: lines 294, 325)
- `src/data/dataset.py` (2 instances: lines 244, 294)

**Fix Pattern:**
```python
# Before:
except Exception as e:
    logger.error(f"Error: {e}")

# After:
except (IOError, ValueError, RuntimeError) as e:
    logger.error(f"Specific error: {e}", exc_info=True)
```

---

### 🟠 3. Evaluation Metric Discrepancy

**Source:** `conductor/archive/eval_export_fixes_20251222/spec.md`

**Issue:** Evaluator reports >99% FNR despite high training F1 (>0.99).

**Root Cause:** Class label misalignment between training and evaluation.

**Actions:**
- [ ] Verify positive/negative class indices are consistent across `training/`, `evaluation/`, `ui/`
- [ ] Confirm feature vectors and labels align between training and evaluation datasets
- [ ] Review FNR and F1 score calculations in `src/evaluation/`
- [ ] Ensure identical preprocessing steps for training and evaluation

---

## P2 - Medium Priority (Performance & Quality)

### 🟡 1. GPU Performance Optimization

**Source:** `conductor/archive/gpu_perf_opt_20251223/spec.md`

**Target:** 2-4x throughput improvement

#### 1.1 Channels Last Memory Format
```python
# Add to trainer.py
self.model = self.model.to(memory_format=torch.channels_last)
```

#### 1.2 Non-Blocking Data Transfers
```python
# In training loop
data = data.to(device, non_blocking=True)
labels = labels.to(device, non_blocking=True)
```

#### 1.3 Persistent DataLoader Workers
```python
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)
```

#### 1.4 Teacher Model GPU Residence
- [ ] Configure Teacher model to load on `cuda` by default
- [ ] Integrate real-time VRAM monitoring into training dashboard

---

### 🟡 2. N+1 Query Pattern in Dataset Scanning

**File:** `src/data/splitter.py` - Line 452-454

**Issue:** `rglob` called for EVERY file - O(n²) complexity.

**Current:**
```python
for npy_file in category_npy_dir.rglob(f"{filename_stem}.npy"):
    return str(npy_file)
```

**Fix:**
```python
def _build_npy_index(self, npy_dir: Path) -> Dict[str, Path]:
    """Pre-index all .npy files - O(n) once"""
    index = {}
    for npy_file in npy_dir.rglob("*.npy"):
        index[npy_file.stem] = npy_file
    return index

# Then: O(1) lookup
npy_path = self.npy_index.get(filename_stem)
```

---

### 🟡 3. Unbounded Feature Cache

**File:** `src/data/dataset.py` - Line 239-240

**Issue:** No cache eviction, can cause OOM for large datasets.

**Fix:**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def _load_cached_npy(self, idx):
    ...
```

---

### 🟡 4. Model Compilation

**File:** `src/training/trainer.py`

**Action:**
```python
if hasattr(torch, 'compile'):
    self.model = torch.compile(self.model, mode="max-autotune")
```

---

### 🟡 5. ONNX/TFLite Export Pipeline

**Source:** `conductor/archive/eval_export_fixes_20251222/spec.md`

**Issues:**
- Conversion failures with QAT models
- "Operator not supported" errors

**Actions:**
- [ ] Fix `onnx2tf` conversion logic
- [ ] Enable reliable TFLite export for QAT models
- [ ] Validate quantized exports maintain performance parity
- [ ] Preserve input/output names and shapes during conversion

---

### 🟡 6. Dynamic CMVN Dimensions

**Source:** `conductor/tracks/dynamic_cmvn_dims_20251223/spec.md`

**Issue:** CMVN stats hardcoded to 40 mel bands.

**Actions:**
- [ ] Remove default `n_mels=40` from `FeatureExtractor`
- [ ] Update model builders for dynamic input dimensions
- [ ] Set 64 mel bands as default in all UI panels
- [ ] Implement CMVN stats dimension mismatch detection
- [ ] Add "Update CMVN Stats" button in UI

---

## P3 - Low Priority (Code Quality & Maintenance)

### 🟢 1. Unused Imports (58 instances)

**Examples:**
```python
# src/data/balanced_sampler.py
import torch  # Unused
from typing import Dict, Optional  # Unused

# src/data/augmentation.py
import numpy as np  # Unused

# src/data/feature_extraction.py
import torchaudio  # Unused
```

**Quick Fix:**
```bash
pip install autoflake
autoflake --remove-all-unused-imports --in-place --recursive src/
```

---

### 🟢 2. File Encoding

**File:** `src/data/file_cache.py` - Lines 40, 52

**Fix:**
```python
# Before:
with open(cache_path, 'r') as f:

# After:
with open(cache_path, 'r', encoding='utf-8') as f:
```

---

### 🟢 3. F-String Placeholders (79 instances)

**Issue:** F-strings without placeholders.

**Example:**
```python
# Wrong:
print(f"This is a message")

# Correct:
print("This is a message")
```

---

### 🟢 4. Import Organization (89 instances)

**Fix:**
```bash
isort src/
```

**Correct Order:**
1. Standard library imports
2. Third-party imports
3. Local module imports

---

### 🟢 5. Line Length (127 instances)

PEP 8 recommends 79-120 characters.

**Fix:**
```bash
black --line-length 120 src/
```

---

### 🟢 6. Magic Numbers

**Example:** `src/data/splitter.py` - Line 55

**Before:**
```python
max_workers = max(multiprocessing.cpu_count() - 2, 1)
```

**After:**
```python
CPU_RESERVE_CORES = 2  # Reserve for OS
max_workers = max(multiprocessing.cpu_count() - CPU_RESERVE_CORES, 1)
```

---

### 🟢 7. Code Duplication

**File:** `src/data/splitter.py` - Lines 545-586

**Issue:** NPY path mapping logic duplicated 3 times.

**Fix:** Extract to helper method `_map_npy_paths()`.

---

### 🟢 8. Mixed Language Comments

**File:** `src/config/validator.py` - Lines 51-89

**Issue:** Mix of Turkish and English comments.

**Fix:** Standardize all comments to English.

---

### 🟢 9. Missing `__all__` Declarations

Add explicit exports to module files:
```python
__all__ = ['WakewordDataset', 'create_dataloader', ...]
```

---

### 🟢 10. Line Endings

**Issue:** Inconsistent LF/CRLF across files.

**Fix:** Add `.gitattributes`:
```gitattributes
* text=auto
*.py text eol=lf
*.md text eol=lf
*.yml text eol=lf
*.yaml text eol=lf
*.json text eol=lf
*.toml text eol=lf
*.txt text eol=lf
*.sh text eol=lf
*.bat text eol=crlf
```

---

## Active Development Tracks

### Track 1: Presets & Pipeline Integration
**File:** `conductor/tracks/presets_pipeline_20251223/spec.md`

**Acceptance Criteria:**
- [ ] `presets.py` contains new parameters for 'Standard' and 'Lightweight' profiles
- [ ] Training logs confirm CMVN is applied
- [ ] Test script verifies streaming hysteresis logic
- [ ] Model size warnings for exceeding RAM/Flash limits
- [ ] Quantization calibration uses representative mix

---

### Track 2: Dynamic CMVN Dimensions
**File:** `conductor/tracks/dynamic_cmvn_dims_20251223/spec.md`

**Acceptance Criteria:**
- [ ] Training with any `n_mels` value works without dimension errors
- [ ] UI prompts for CMVN recomputation on dimension change
- [ ] Models initialize with correct input shapes
- [ ] "40 Mel" hardcoding removed from codebase

---

### Track 3: Config Sync & ESP32 Optimization
**File:** `conductor/tracks/config_sync_esp32_opt_20251223/spec.md`

**Acceptance Criteria:**
- [ ] Mismatch warning on loading incompatible configs
- [ ] `tiny_conv` profile includes distillation and augmentation
- [ ] Models fit on ESP32-S3 target
- [ ] Exported models in ESPHome-compatible path

---

### Track 4: Startup & HPO Fixes
**File:** `conductor/tracks/startup_hpo_fixes_20251223/spec.md`

**Acceptance Criteria:**
- [ ] `python run.py` reaches interactive Gradio UI
- [ ] HPO trial completes without crashing
- [ ] `best_f1` tracked correctly across trials

---

## Testing Checklist

### Panel 2 Configuration Tests
- [ ] Load preset → verify RIR and NPY parameters populate correctly
- [ ] Adjust RIR dry/wet sliders → save config → reload → verify persistence
- [ ] Toggle NPY checkboxes → save config → reload → verify persistence
- [ ] Validate configuration → verify no errors with new parameters
- [ ] Reset to defaults → verify all parameters reset correctly

### Panel 1 Batch Extraction Tests
- [ ] Scan dataset → click batch extract → verify error message
- [ ] Scan dataset → configure extraction → click extract → verify progress bar
- [ ] Check extraction log → verify detailed report with next steps
- [ ] Verify .npy files created in output directory with correct structure
- [ ] Analyze existing NPY → verify report shows correct statistics

### End-to-End Workflow Tests
- [ ] Extract NPY features → enable in config → start training → verify 40-60% speedup
- [ ] Configure RIR dry/wet → train → verify reverberation applied correctly
- [ ] Use both features together → verify no conflicts

### CI/CD Checklist
- [ ] All tests passing
- [ ] Coverage >80%
- [ ] No linting errors
- [ ] Mobile testing complete (if applicable)
- [ ] Environment variables configured
- [ ] Database migrations ready (if applicable)
- [ ] Backup created

---

## Timeline Estimate

| Phase | Duration | Tasks |
|-------|----------|-------|
| **Phase 1: Critical Fixes** | 1-2 days | P0 undefined names, startup bugs |
| **Phase 2: Security** | 1 day | P1 torch.load, hash, path traversal |
| **Phase 3: Performance** | 2-3 days | P2 GPU optimization, caching |
| **Phase 4: Code Quality** | 1-2 weeks | P3 cleanup, documentation |
| **Total Estimate** | 2-3 weeks | All tasks |

---

## Quick Fix Scripts

### Remove Unused Imports
```bash
pip install autoflake
autoflake --remove-all-unused-imports --in-place --recursive src/
```

### Fix Import Organization
```bash
isort src/
```

### Format Code
```bash
black --line-length 120 src/
```

### Run All Checks
```bash
pylint src/ --exit-zero
pyflakes src/
bandit -r src/ -ll
```

### Pre-commit Setup
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
```

---

## Document History

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-23 | 1.0.0 | Initial comprehensive action plan |

---

*This action plan was generated by consolidating all comments, suggestions, and action items from:*
- `DEVELOPMENT_ROADMAP.md` (CODE_ANALYSIS_REPORT, CODE_REVIEW_REPORT, wakeword_project_analysis_report)
- `conductor/tracks/` (4 active tracks)
- `conductor/archive/` (12 archived tracks)
- Source code comments (TODO, FIXME, CRITICAL, BUG markers)
