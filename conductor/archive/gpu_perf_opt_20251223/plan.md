# Implementation Plan - High-Performance GPU Training Pipeline

## Phase 1: Core Performance Optimizations (Zero-Impact) [checkpoint: completed]
Implement optimizations that improve throughput without affecting model accuracy.

- [x] **Task 1: Implement `channels_last` Memory Format**
    - [x] Update `src/training/trainer.py` to convert models to `channels_last` on GPU.
    - [x] Update `src/training/training_loop.py` to ensure input batches use the same format.
    - [x] **TDD:** Write a test to verify that `model.parameters()` and input tensors use `channels_last` when on `cuda`.
- [x] **Task 2: Implement Async Data Transfers (`non_blocking`)**
    - [x] Update `src/training/training_loop.py` to use `non_blocking=True` in all `.to(device)` calls for inputs, targets, and metadata.
    - [x] **TDD:** Verify that training execution remains stable with non-blocking transfers enabled.
- [x] **Task 3: Enable Persistent Workers**
    - [x] Update DataLoader initialization in `src/ui/panel_training.py` and `src/training/hpo.py` to set `persistent_workers=True`.
    - [x] **TDD:** Verify that DataLoaders do not shutdown workers between epochs.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Core Performance' (Protocol in workflow.md)

## Phase 2: GPU Device Strategy & Teacher Migration [checkpoint: completed]
Move computational load to the GPU and optimize device placement.

- [x] **Task 1: Migrate Distillation Teacher to GPU**
    - [x] Update `src/training/distillation_trainer.py` to ignore `teacher_on_cpu` setting or default it to `False`.
    - [x] Ensure Teacher loading happens within a `torch.no_grad()` block to minimize peak memory during model initialization.
    - [x] **TDD:** Write a test checking that `trainer.teacher.device` is `cuda` even when distillation is active.
- [x] **Task 2: Implement Pre-flight VRAM Estimation**
    - [x] Add a utility in `src/config/cuda_utils.py` to estimate VRAM usage based on Teacher architecture and Batch Size.
    - [x] Log this estimate at the start of training in `src/ui/panel_training.py`.
- [x] Task: Conductor - User Manual Verification 'Phase 2: GPU Device Strategy' (Protocol in workflow.md)

## Phase 3: Telemetry & Safety [checkpoint: completed]
Enhance the UI for memory monitoring and implement crash protection.

- [x] **Task 1: Integrate Real-time VRAM Telemetry**
    - [x] Update `get_training_status` in `src/ui/panel_training.py` to query and return active VRAM usage.
    - [x] Add a "VRAM Usage" gauge or number component to the Training UI.
- [x] **Task 2: Implement Graceful OOM Recovery**
    - [x] Wrap the main training loop in a try-except block specifically for `RuntimeError` matching "out of memory".
    - [x] Return a user-friendly error message to the Gradio status box with instructions to lower batch size.
- [x] **Task 3: Performance Benchmarking**
    - [x] Conduct a benchmark run with Distillation ON to quantify samples/sec gain.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Telemetry & Safety' (Protocol in workflow.md)