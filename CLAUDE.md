# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Wakeword Training Platform** - A production-ready, GPU-accelerated platform for training custom wakeword detection models (e.g., "Hey Siri", "Alexa"). Built with PyTorch and Gradio, featuring advanced training optimizations (CMVN, EMA, Mixed Precision) and production-grade metrics (FAH, EER).

**Version**: 2.0.0
**Python**: 3.8+
**CUDA**: 11.8+ (strict GPU requirement)

## Quick Start Commands

### Launch Application
```bash
python run.py
```
Opens Gradio web interface at `http://localhost:7860`

### Install Dependencies
```bash
# For CUDA 11.8 (GPU - Recommended)
pip install torch==2.1.2+cu118 torchaudio==2.1.2+cu118 torchvision==0.16.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# For CPU only (slow, not recommended)
pip install -r requirements.txt
```

### Run Tests
No test suite exists. When modifying core logic, create standalone verification scripts in `docs/examples/` or temporary test files.

## Architecture

### High-Level System Design

**Entry Point**: `run.py` → `src/ui/app.py` (Gradio interface)

**6-Panel Workflow**:
1. **Dataset** → Scan audio, split data, precompute features
2. **Configuration** → Model selection, hyperparameters, feature toggles
3. **Training** → GPU-accelerated training loop with real-time metrics
4. **Evaluation** → File/microphone testing, advanced metrics (FAH, EER)
5. **Export** → ONNX/TorchScript/Quantized model export
6. **Documentation** → In-app help and guides

### Core Module Structure

```
src/
├── config/          # Pydantic-based configuration management
│   ├── defaults.py         # Default hyperparameters (START HERE for configs)
│   ├── pydantic_validator.py  # Strict validation schemas
│   └── validator.py        # Legacy config handler
│
├── data/            # Data pipeline (audio → features → batches)
│   ├── dataset.py          # WakewordDataset: audio loading, caching
│   ├── cmvn.py             # Cepstral Mean Variance Normalization
│   ├── augmentation.py     # Time stretch, pitch shift, noise, RIR
│   ├── feature_extraction.py  # Mel spectrogram extraction
│   ├── file_cache.py       # LRU caching for .npy features
│   └── balanced_sampler.py # Maintain class ratios in batches
│
├── models/          # Model architectures
│   ├── architectures.py    # ResNet18/34, MobileNetV3, LSTM, GRU, TCN
│   ├── losses.py           # Focal Loss, Label Smoothing, ArcFace
│   └── temperature_scaling.py  # Post-training calibration
│
├── training/        # Training loop and optimizations
│   ├── trainer.py          # Main training loop (AMP, Gradient Clipping)
│   ├── training_loop.py    # Epoch/batch iteration logic
│   ├── ema.py              # Exponential Moving Average for weights
│   ├── lr_finder.py        # Automated learning rate discovery
│   ├── optimizer_factory.py # AdamW, SGD, AdaBound creation
│   ├── metrics.py          # Accuracy, Precision, Recall, F1
│   └── checkpoint_manager.py # Save/load best models
│
├── evaluation/      # Inference and metrics
│   ├── evaluator.py        # Basic metrics (accuracy, F1)
│   ├── advanced_evaluator.py  # FAH, EER, pAUC, ROC curves
│   ├── streaming_detector.py  # Real-time detection with voting
│   └── inference.py        # Single-file inference
│
├── export/          # Model deployment
│   └── onnx_exporter.py    # ONNX/TorchScript/Quantized export
│
└── ui/              # Gradio interface (pure presentation layer)
    ├── app.py              # Main interface assembly
    ├── panel_dataset.py    # Dataset management panel
    ├── panel_config.py     # Configuration panel
    ├── panel_training.py   # Training panel with live plots
    ├── panel_evaluation.py # Evaluation panel
    └── panel_export.py     # Export panel
```

### Data Flow

```
Raw Audio (data/raw/)
  → Feature Extraction (Mel spectrogram)
  → CMVN Normalization
  → Augmentation (train only)
  → Batching (Balanced Sampler)
  → GPU Training
  → EMA Weight Update
  → Checkpoint (best_model.pt)
  → ONNX Export
```

### Configuration Management

**Critical Pattern**: Always use `src.config` classes. **DO NOT hardcode hyperparameters.**

```python
# ✓ CORRECT
from src.config.defaults import get_default_config
config = get_default_config()
lr = config.training.learning_rate

# ✗ WRONG
lr = 0.001  # Hardcoded value
```

**Configuration Files**:
- `src/config/defaults.py` - Default values (DataConfig, TrainingConfig, ModelConfig, etc.)
- `src/config/pydantic_validator.py` - Pydantic schemas for validation
- `configs/*.yaml` - User-saved configurations (generated via UI)

### Key Technical Concepts

#### CMVN (Cepstral Mean Variance Normalization)
- **Purpose**: Normalize features across dataset for consistent acoustic representations
- **Location**: `src/data/cmvn.py`
- **Impact**: +2-4% accuracy, better cross-device performance
- **Stats Storage**: `data/cmvn_stats.json`
- **When to Use**: Always enabled for production models
- **See**: `TECHNICAL_FEATURES.md` Section 1.1

#### EMA (Exponential Moving Average)
- **Purpose**: Create smoother, more stable model by averaging weights over time
- **Location**: `src/training/ema.py`
- **Impact**: +1-2% validation accuracy, more consistent predictions
- **Decay**: 0.999 (default)
- **Integration**: Automatically applied in `Trainer` if `config.training.use_ema=True`

#### Balanced Sampling
- **Purpose**: Handle class imbalance (typically 2-5× more negatives)
- **Location**: `src/data/balanced_sampler.py`
- **Impact**: 20-30% faster convergence, 5-15% reduction in false positives
- **When to Use**: Enable when negatives > 2× positives

#### FAH (False Alarms per Hour)
- **Purpose**: Production metric for user experience
- **Location**: `src/evaluation/advanced_evaluator.py`
- **Formula**: `FAH = (FP / total_seconds) × 3600`
- **Target**: ≤1.0 for balanced use cases
- **See**: `TECHNICAL_FEATURES.md` Section 4.2

#### LR Finder
- **Purpose**: Automatically find optimal learning rate
- **Location**: `src/training/lr_finder.py`
- **Algorithm**: Exponential range test (Leslie Smith)
- **When to Use**: First training or significant data changes
- **Time Cost**: 2-5 minutes

## Development Guidelines

### Coding Conventions

1. **Type Hinting**: Enforce strict type hints, especially for Pydantic models
   ```python
   from pathlib import Path
   from typing import Optional, Tuple

   def load_audio(path: Path) -> Tuple[np.ndarray, int]:
       ...
   ```

2. **Path Handling**: Use `pathlib.Path` for ALL file operations
   ```python
   # ✓ CORRECT
   from pathlib import Path
   data_dir = Path("data/raw")
   audio_file = data_dir / "positive" / "sample.wav"

   # ✗ WRONG
   audio_file = "data/raw/positive/sample.wav"
   ```

3. **Logging**: Use structured logger, not print()
   ```python
   from src.config.logger import setup_logger
   logger = setup_logger(__name__)

   logger.info("Training started", epoch=1, lr=0.001)
   logger.error("Failed to load audio", path=str(audio_path), exc_info=True)
   ```

4. **GPU Optimizations**: Respect configuration flags
   ```python
   # Mixed Precision
   if config.optimizer.mixed_precision:
       with torch.cuda.amp.autocast():
           output = model(input)

   # EMA Updates
   if config.training.use_ema and self.ema is not None:
       self.ema.update()
   ```

### Critical Implementation Patterns

#### Loading Trained Models
```python
# Load checkpoint with proper device mapping
checkpoint = torch.load("models/checkpoints/best_model.pt", map_location="cuda")
model.load_state_dict(checkpoint["model_state_dict"])

# If EMA was used during training
if "ema_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["ema_state_dict"])  # Use EMA weights
```

#### Dataset Structure
```
data/
├── raw/                    # User-provided audio files
│   ├── positive/          # Wakeword samples (500-2000)
│   └── negative/          # Non-wakeword samples (1000-5000)
├── splits/                # Auto-generated train/val/test manifests
│   └── split_*.json
├── npy/                   # Precomputed features (optional cache)
└── cmvn_stats.json       # CMVN normalization stats
```

#### Feature Extraction Pipeline
```python
# CPU-based extraction → GPU-based training
from src.data.feature_extraction import FeatureExtractor

extractor = FeatureExtractor(
    feature_type="mel",
    sample_rate=16000,
    n_fft=512,
    hop_length=160,
    n_mels=128
)
features = extractor.extract(audio)  # Shape: (n_mels, time_steps)
```

### Common Pitfalls

1. **DO NOT** modify `.npy` files directly - regenerate via UI or `src/data/npy_extractor.py`
2. **DO NOT** train without CMVN enabled for production models
3. **DO NOT** use `fit()` method on models - use `Trainer` class
4. **DO NOT** hardcode paths - use `config.paths.*`
5. **DO NOT** assume CPU/MPS training support - **CUDA ONLY**

### Testing Strategy

**No formal test suite exists.** When modifying core components:

1. Create standalone reproduction script:
   ```python
   # docs/examples/test_my_feature.py
   from src.training.trainer import Trainer
   from src.config.defaults import get_default_config

   config = get_default_config()
   trainer = Trainer(model, config)
   # ... test logic
   ```

2. Test via UI for integration checks:
   - Dataset panel: Verify data loading
   - Training panel: Run 2-3 epochs with small data
   - Evaluation panel: Check metrics computation

3. Visual inspection of outputs (plots, logs, checkpoints)

## Important Files

- **`TECHNICAL_FEATURES.md`**: **MUST READ** for understanding CMVN, EMA, FAH math/logic
- **`README.md`**: User-facing quick start guide
- **`.github/copilot-instructions.md`**: Additional AI agent guidelines (imported above)
- **`src/config/defaults.py`**: Single source of truth for default hyperparameters
- **`src/training/trainer.py`**: Main training logic (150+ lines)
- **`src/evaluation/advanced_evaluator.py`**: Production metrics implementation

## Known Issues & Constraints

1. **GPU Requirement**: Strict NVIDIA CUDA requirement. No CPU/MPS training support.
2. **"God Object" Classes**: `Trainer` and `Dataset` classes need refactoring (see `docs/IMPROVEMENT_PLAN.md`)
3. **No Formal Tests**: Rely on manual testing and UI validation
4. **Pydantic Migration**: Partially complete - some modules still use legacy config
5. **Dataset Logic**: Dynamic label mapping and fallback logic needs refactoring (see `TECHNICAL_DESIGN_DATA_REFACTOR.md`)

## Deployment Notes

**Model Export**: Use `src/export/onnx_exporter.py` for production deployment
```bash
# Via UI: Panel 5 → Export Model
# Formats: ONNX (universal), TorchScript (PyTorch), INT8 (quantized)
```

**Production Checklist**:
- ✓ CMVN enabled during training
- ✓ EMA used for final weights
- ✓ Temperature scaling applied (Panel 4)
- ✓ Streaming detection with voting (not single-frame)
- ✓ Target FAH ≤ 1.0 validated on test set

## Additional Resources

- **Serena Memory**: `project_context` memory contains development roadmap
- **SuperClaude Framework**: See `~/.claude/superclaude/` for custom AI agent instructions
- **Experiment Tracking**: WandB integration available (see `src/training/wandb_callback.py`)
