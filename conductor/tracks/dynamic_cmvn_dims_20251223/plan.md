# Implementation Plan: Dynamic CMVN Dimensions & Hardcoding Removal

## Phase 1: Core Logic & Hardcoding Cleanup
Refactor the underlying feature extraction and model architecture logic to eliminate hardcoded dimension defaults.

- [x] **Task 1.1: Feature Extraction Cleanup**
    - Refactor `src/data/feature_extraction.py` to remove `n_mels=40` default. [done: 2025-12-23]
- [x] **Task 1.2: Model Architecture Refactor**
    - Audit `src/models/architectures.py` and remove hardcoded `input_size=40` or `input_channels=40` in all model classes (LSTM, GRU, TCN, ResNet, etc.). [done: 2025-12-23]
- [x] **Task 1.3: Centralized Defaults Alignment**
    - Verify `src/config/defaults.py` and ensure `n_mels=64` is the source of truth for all modules.
    - Update `src/ui/panel_config.py` and `src/ui/panel_dataset.py` to use `64` as the UI default. [done: 2025-12-23]
- [x] **Task: Conductor - User Manual Verification 'Phase 1: Core Logic & Hardcoding Cleanup' (Protocol in workflow.md)** [done: 2025-12-23]

## Phase 2: CMVN Mismatch & UI Interaction
Implement the user-triggered recomputation workflow and enhance UI feedback.

- [x] **Task 2.1: Mismatch Detection Enhancement**
    - Update `AudioProcessor` in `src/data/processor.py` to expose a mismatch flag instead of just logging a warning. [done: 2025-12-23]
- [x] **Task 2.2: UI Warning & Recomputation Button**
    - Add a "Recompute CMVN Stats" button to `src/ui/panel_training.py` that becomes active/highlighted when a mismatch is detected.
    - Implement a "Soft Warning" popup or status message when training is initiated with a mismatch. [done: 2025-12-23]
- [x] **Task 2.3: Consistency Check in Presets**
    - Review `src/config/presets.py` to ensure all presets explicitly define their dimensions and that switching between them correctly triggers the UI mismatch warning. [done: 2025-12-23]
- [x] **Task: Conductor - User Manual Verification 'Phase 2: CMVN Mismatch & UI Interaction' (Protocol in workflow.md)** [done: 2025-12-23]

## Phase 3: Integration & Verification
Ensure the end-to-end pipeline works for various mel band configurations.

- [x] **Task 3.1: Unit Testing for Dynamic Dimensions**
    - Create a test script `tests/test_dynamic_dimensions.py` that verifies model initialization and feature extraction for 40, 64, and 80 mel bands. [done: 2025-12-23]
- [x] **Task 3.2: E2E Verification**
    - Verify that extracting NPY features at 80 mels and then training with 80 mels works without "hardcoded 40" errors.
    - Confirm the "Soft Warning" appears if stats are at 40 and config is at 64. [done: 2025-12-23]
- [x] **Task: Conductor - User Manual Verification 'Phase 3: Integration & Verification' (Protocol in workflow.md)** [done: 2025-12-23]