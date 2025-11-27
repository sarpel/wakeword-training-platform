# Wakeword Training Platform - AI Agent Instructions

## Project Overview
This is a production-ready Wakeword Detection Training Platform using **PyTorch** for the backend and **Gradio** for the web UI. It features GPU acceleration, advanced training optimizations (CMVN, EMA, Mixed Precision), and production-grade metrics (FAH, EER).

## Architecture & Core Components
- **Entry Point**: `run.py` launches the Gradio interface (`src/ui/app.py`).
- **UI**: `src/ui/` contains Gradio panels (`panel_dataset.py`, `panel_training.py`, etc.).
- **Data Pipeline** (`src/data/`):
  - `dataset.py`: Handles audio loading and caching.
  - `cmvn.py`: Cepstral Mean and Variance Normalization.
  - `augmentation.py`: Time stretch, pitch shift, noise, RIR.
  - `file_cache.py`: LRU caching for `.npy` features.
- **Training** (`src/training/`):
  - `trainer.py`: Custom training loop with AMP and Gradient Clipping.
  - `ema.py`: Exponential Moving Average for model weights.
  - `lr_finder.py`: Automated learning rate discovery.
- **Configuration** (`src/config/`):
  - Uses **Pydantic** models for validation (`pydantic_validator.py`) and **YAML** for persistence.
- **Evaluation** (`src/evaluation/`):
  - `advanced_metrics.py`: Calculates FAH (False Alarms/Hour), EER, pAUC.
  - `streaming_detector.py`: Real-time detection logic with voting and hysteresis.

## Critical Workflows
- **Start Application**: `python run.py` (opens http://localhost:7860).
- **Data Structure**:
  - Raw audio: `data/raw/positive/` and `data/raw/negative/`.
  - Precomputed features: `data/npy/` (generated via UI or scripts).
  - Splits: `data/splits/` (JSON manifests).
- **Training**:
  - Configured via UI, saved to `configs/`.
  - Checkpoints saved to `models/checkpoints/`.
  - Best model is always `best_model.pt`.

## Coding Conventions & Patterns
- **Configuration**: Always use `src.config` classes. Do not hardcode hyperparameters.
  - Example: `config.training.learning_rate` instead of raw values.
- **Logging**: Use the project's structured logger.
  - `from src.config.logger import setup_logger; logger = setup_logger(__name__)`
- **Path Handling**: Use `pathlib.Path` for all file operations.
- **Type Hinting**: Enforce strict type hints, especially for Pydantic models.
- **Optimizations**:
  - Respect `config.optimizer.mixed_precision` (use `torch.cuda.amp`).
  - Implement `EMA` updates in training loops if enabled.

## Development Guidelines
- **Testing**: No test suite exists (`tests/` is missing).
  - **Requirement**: When modifying core logic, create a standalone reproduction script in `docs/examples/` or a temporary test file to verify changes.
- **Dependencies**: Strict versioning in `requirements.txt` (PyTorch 2.1.2+cu118).
- **Model Export**: Use `src/export/onnx_exporter.py` for deployment artifacts.

## Key Files
- `TECHNICAL_FEATURES.md`: **Must read** for understanding math/logic behind CMVN, EMA, and FAH.
- `src/config/defaults.py`: Default hyperparameter values.
- `src/training/trainer.py`: Main training logic.
