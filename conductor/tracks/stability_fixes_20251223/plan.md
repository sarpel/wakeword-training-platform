# Implementation Plan: Terminal Output & Training Stability Fixes

## Phase 1: Metric Stability and UI Fixes
- [x] Task: Fix `KeyError: 'val_eer'` in Training UI [4d7a418]
    - [x] Sub-task: Write TDD test to simulate missing metrics in `src/ui/panel_training.py`
    - [x] Sub-task: Implement defensive history initialization and `.get()` access in `TrainingCallback`
- [x] Task: Guarantee Metric Consistency in Training Loop [4d7a418]
    - [x] Sub-task: Write TDD test for `src/training/training_loop.py` to verify metric dictionary integrity
    - [x] Sub-task: Update validation logic to ensure `val_eer` is always calculated or defaulted
- [ ] Task: Conductor - User Manual Verification 'Metric Stability and UI Fixes' (Protocol in workflow.md)

## Phase 2: Initialization & Logging Cleanup
- [ ] Task: Silence Wav2Vec2 Initialization Warnings
    - [ ] Sub-task: Research and implement HuggingFace `logging` overrides to suppress "newly initialized" warnings
- [ ] Task: Standardize Teacher Loading Logs
    - [ ] Sub-task: Replace verbose print/log statements with a single clean status message
- [ ] Task: Conductor - User Manual Verification 'Initialization & Logging Cleanup' (Protocol in workflow.md)

## Phase 3: Observability Enhancements
- [ ] Task: Integrate WandB Weave for Advanced Tracing
    - [ ] Sub-task: Implement `weave.init()` in the training initialization pipeline
    - [ ] Sub-task: Verify that terminal suggestions for Weave are silenced
- [ ] Task: Conductor - User Manual Verification 'Observability Enhancements' (Protocol in workflow.md)
