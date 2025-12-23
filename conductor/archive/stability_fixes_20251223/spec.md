# Track Specification: Terminal Output & Training Stability Fixes

## Overview
This track addresses a critical training crash related to missing metrics, resolves initialization warnings in the teacher model loading phase, and integrates recommended developer tools (WandB Weave) to clean up terminal output and improve observability.

## Functional Requirements

### 1. Training Stability (`val_eer` fix)
- **Source Validation:** Ensure the training loop (`src/training/training_loop.py`) always computes and returns a complete `val_metrics` dictionary, including `val_eer`.
- **Defensive UI:** Update `src/ui/panel_training.py` to use `.get()` or a defensive initialization pattern for the `history` dictionary to prevent `KeyError`.
- **Error Resilience:** Implement a fallback mechanism where missing metrics are logged as warnings rather than stopping the training session.

### 2. Teacher Model Initialization
- **Warning Suppression:** Resolve or suppress the "weights not initialized" warnings for `Wav2Vec2Model` by ensuring the loading configuration is optimized for inference/distillation.
- **Log Polishing:** Clean up the teacher loading logs to provide a clear, successful initialization status.

### 3. Observability Integration
- **WandB Weave:** Integrate `import weave` into the training pipeline as suggested by the WandB terminal output to enable advanced call tracing and visualization.
- **Environment Parity:** Ensure the integration works seamlessly in the current Win32 environment.

## Non-Functional Requirements
- **Terminal Readability:** Reduce overall log noise by 30% by standardizing info/warning levels.
- **Maintainability:** Use a consistent metric key mapping between the trainer and the UI.

## Acceptance Criteria
- [ ] Training successfully completes Epoch 1 and continues to Epoch 2 without a `KeyError`.
- [ ] Terminal output no longer shows "Some weights of Wav2Vec2Model were not initialized".
- [ ] WandB logs confirm that Weave is initialized and tracking the run.
- [ ] No "import weave" suggestions appear in the console.

## Out of Scope
- Refactoring the entire metrics calculation engine.
- Adding new deep learning architectures.
