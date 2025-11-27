# Action Plan for PR #1 (Revised)

After verifying the codebase, the following issues are confirmed to be active bugs that need to be fixed. Other issues mentioned in the initial review (missing imports in panels, checkpoint manager, etc.) were found to be **already fixed**.

## 1. Critical Logic Errors (Must Fix)

### `src/data/splitter.py`
- **Issue**: The `DatasetScanner.scan_dataset` method expects `validate_audio_file` to return a tuple `(is_valid, metadata, error)`, but the `AudioValidator` now raises exceptions instead of returning status.
- **Fix**: Wrap the validation call in a `try-except` block.
  ```python
  # Current (Broken):
  is_valid, metadata, error = self.validator.validate_audio_file(file_path)
  
  # Fix:
  try:
      metadata = self.validator.validate_audio_file(file_path)
      is_valid = True
      error = None
  except DataLoadError as e:
      is_valid = False
      metadata = None
      error = str(e)
  ```

### `src/training/wandb_callback.py`
- **Issue**: The `WandbCallback.on_epoch_end` method attempts to access `val_metrics` as a dictionary (e.g., `val_metrics.get("accuracy")`), but it is passed as a `MetricResults` object.
- **Fix**: Access attributes directly or convert to dict.
  ```python
  # Fix:
  val_acc = val_metrics.accuracy
  val_f1 = val_metrics.f1_score
  # ...
  ```
- **Issue**: Missing `finish()` method to close the run.
- **Fix**: Add `finish()` method calling `wandb.finish()`.

### `src/ui/panel_evaluation.py`
- **Issue**: The lambda function for the "Evaluate" button crashes if `config_state` is None or empty.
- **Fix**: Add a guard clause or default value.

## 2. Memory & Performance

### `src/data/dataset.py`
- **Issue**: `@lru_cache` is used on the instance method `_load_from_npy`. Since `self` is part of the cache key, this keeps `WakewordDataset` instances alive, causing memory leaks during HPO or repeated training runs.
- **Fix**: Remove `@lru_cache` or implement a custom caching mechanism that doesn't bind to `self` in a way that prevents GC (or just rely on the OS page cache for memory mapping).

## 3. Deprecated APIs & Stability

### `src/training/training_loop.py`
- **Issue**: Uses deprecated `torch.cuda.amp.autocast`.
- **Fix**: Update to `torch.amp.autocast("cuda", ...)` for PyTorch 2.x compatibility.

### `src/export/onnx_exporter.py`
- **Issue**: Calls `torch.cuda.synchronize()` unconditionally, which might fail on CPU-only environments (though the project enforces CUDA, it's good practice to guard it).
- **Fix**: Add `if torch.cuda.is_available():` check.

## 4. Verification Steps
After applying these fixes, run:
1. `python src/ui/app.py` and try to scan a dataset (tests `splitter.py`).
2. Run a short training session (tests `wandb_callback.py` and `training_loop.py`).
