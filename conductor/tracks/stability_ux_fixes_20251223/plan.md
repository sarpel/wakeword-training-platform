# Implementation Plan - Stability & UX Enhancement Suite

## Phase 1: Infrastructure & Preset Updates [checkpoint: 0ce6044]
Update existing configuration presets and suppress terminal noise to establish a stable baseline.

- [x] Task: Update `tiny_conv (esp32s3)` preset defaults in configuration files. [b3ab1d3]
- [x] Task: Implement `ConnectionResetError` suppression in the server/async logic. [b389daf]
- [x] Task: Conductor - User Manual Verification 'Infrastructure & Preset Updates' (Protocol in workflow.md) [0ce6044]

## Phase 2: QAT Report Stability
Fix the channel mismatch error during quantization reporting.

- [ ] Task: Write failing test to reproduce QAT channel mismatch (`tests/reproduce_qat_mismatch.py`).
- [ ] Task: Fix model/input dimension alignment in `src/training/trainer.py` for QAT reporting.
- [ ] Task: Verify QAT report generation works for both HPO trial ends and early stopping.
- [ ] Task: Conductor - User Manual Verification 'QAT Report Stability' (Protocol in workflow.md)

## Phase 3: HPO Profile Persistence
Implement the "Load Latest" functionality in the UI.

- [ ] Task: Write unit tests for `ConfigManager` to load the most recent HPO profile from disk.
- [ ] Task: Add "Load Latest HPO Profile" button to the Gradio UI (`src/ui/panels/training.py`).
- [ ] Task: Connect UI button to the backend loading logic and ensure state synchronization.
- [ ] Task: Conductor - User Manual Verification 'HPO Profile Persistence' (Protocol in workflow.md)
