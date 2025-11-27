# Action Plan for PR #1 (Gemini3)

This plan outlines the necessary fixes and improvements based on the code review of Pull Request #1. The items are categorized by priority.

## 🚨 Critical Priority (Bugs, Crashes, Missing Imports)

These issues cause runtime errors or immediate failures and must be fixed first.

### Missing Imports & NameErrors
- **`src/data/splitter.py`**: Fix `validate_audio_file` call. The function now raises `DataLoadError` but the code expects a tuple `(is_valid, metadata, error)`. Wrap in `try-except DataLoadError`.
- **`src/training/wandb_callback.py`**: Fix `val_metrics` access. It receives a `MetricResults` object, not a dict. Change `.get("accuracy")` to `.accuracy`, etc.
- **`src/export/onnx_exporter.py`**: Add missing imports: `onnx`, `onnxruntime as ort`, `numpy as np`. Handle optional dependencies.
- **`src/ui/panel_evaluation.py`**: Add missing imports: `WakewordDataset`, `SimulatedMicrophoneInference`.
- **`src/ui/panel_training.py`**: Add missing import: `MetricResults`.
- **`src/training/checkpoint_manager.py`**: Add missing imports: `json`, `shutil`. Use `TYPE_CHECKING` for `Trainer` type hint to avoid circular imports/runtime errors.
- **`src/data/dataset.py`**: Fix undefined `splits_dir` variable. Change to `data_root / "splits"`.

### Deprecated APIs (Must Migrate)
- **PyTorch AMP**: Replace deprecated `torch.cuda.amp.autocast` with `torch.amp.autocast("cuda")` in:
    - `src/evaluation/inference.py`
    - `src/evaluation/advanced_evaluator.py`
    - `src/evaluation/file_evaluator.py`
    - `src/training/training_loop.py`
- **Pydantic**: Migrate from V1 API (`parse_obj`, `@validator`) to V2 API (`model_validate`, `@field_validator`) in `src/config/pydantic_validator.py`.

## 🛠️ High Priority (Stability, Logic Fixes)

These issues affect the correctness or stability of the application.

### Logic & Stability
- **`src/data/cmvn.py`**: Add check for empty `features_list` to avoid `TypeError` when computing mean.
- **`src/training/training_loop.py`**: Add zero-division check for `num_batches` when calculating `avg_loss`.
- **`src/training/checkpoint.py`**: Use `.get('scaler_state_dict')` instead of direct access to support older checkpoints.
- **`src/export/onnx_exporter.py`**:
    - Guard `torch.cuda.synchronize()` calls to run only if device is CUDA.
    - Fix `QuantType.QFloat16` usage (use `convert_float_to_float16`).
- **`src/training/hpo.py`**:
    - Remove hardcoded `device='cuda'`. Use `cuda` if available, else `cpu`.
    - Catch `optuna.exceptions.OptunaError` specifically.
- **`src/ui/panel_evaluation.py`**: Fix unsafe lambda access to `config_state.get('config')`.
- **`src/training/trainer.py`**:
    - Fix `resume_from` logic: Update `self.state.epoch` and `global_step` from the loaded checkpoint so training doesn't restart at epoch 1.
    - Update test code at the bottom to use `CheckpointManager` instead of `checkpoint_dir`.
- **`src/training/wandb_callback.py`**:
    - Add `wandb.finish()` method to properly close runs.
    - Add `step=batch_idx` to `wandb.log` calls.
- **`src/data/dataset.py`**: Remove `@lru_cache` from instance method `_load_from_npy` to prevent memory leaks.
- **`src/evaluation/streaming_detector.py`**: `process_audio_stream` has a placeholder score `0.0`. Implement actual inference or mark as TODO.

### Configuration & Validation
- **`src/config/pydantic_validator.py`**:
    - Fix `n_fft_must_be_power_of_two` validator (currently does nothing).
    - Add `None` checks in `min_less_than_max` validator.
- **`src/config/validator.py`**:
    - Improve GPU memory estimation (use `torch.cuda.mem_get_info`).
    - Include feature size (mel spectrograms) in memory calculation, not just raw audio.

## 🧹 Medium Priority (Code Quality, Cleanup)

Improvements to code health, logging, and removing dead code.

### Unused Code & Imports
- **Remove Unused Imports**:
    - `src/data/dataset.py`: `create_balanced_sampler_from_dataset`, `compute_cmvn_from_dataset`.
    - `src/training/trainer.py`: `_save_checkpoint`, `load_checkpoint`.
    - `src/training/hpo.py`: `logging`.
    - `src/evaluation/advanced_evaluator.py`: `List`.
    - `src/ui/app.py`: `structlog`.
    - `src/data/splitter.py`: `logging`.
- **Remove Unused Variables**:
    - `src/ui/panel_training.py`: `test_ds`.
    - `src/ui/panel_dataset.py`: `validate_shape_checkbox`, `splits`.
    - `src/data/balanced_sampler.py`: `drop_last` (or implement it).
    - `src/evaluation/inference.py`: `device` in `SimulatedMicrophoneInference` (or use it).

### Logging Improvements
- Use `logger.exception` instead of `logger.error` inside `except` blocks for better stack traces in:
    - `src/training/hpo.py`
    - `src/data/npy_extractor.py`
    - `src/ui/panel_dataset.py`
    - `src/ui/panel_config.py`
    - `src/ui/panel_evaluation.py` (remove double logging).
- Replace `print` with `logger` in `src/data/splitter.py`.

## 📝 Low Priority (Documentation, Nitpicks)

- **Docstrings**: Update docstrings in `src/data/audio_utils.py`, `src/data/feature_extraction.py`, `src/training/trainer.py`, `src/data/splitter.py` to match implementation.
- **Scripts**: Add `venv` checks to `start_app.sh` and `start_app.bat`.
- **Formatting**: Fix indentation in `src/data/audio_utils.py`.
- **Type Hints**: Update return types in `src/evaluation/advanced_evaluator.py` (`Optional[ThresholdMetrics]`).
