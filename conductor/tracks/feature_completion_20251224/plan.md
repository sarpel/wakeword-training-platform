# Implementation Plan: Feature Audit & Full Pipeline Completion

This plan integrates the missing "Google-tier" features (Distributed Cascade, Background Mining, SNR Scheduling) into the UI and core training loop using a TDD approach.

## Phase 1: Training Loop & Optimization Enhancements
Focus on enabling high-performance features and robust training scheduling.

- [x] Task: Implement SNR Scheduling logic in `AudioAugmentation` and `Trainer` e379587
    - Create `tests/test_snr_scheduling.py` to verify SNR decreases over epochs.
    - Update `AudioAugmentation.set_epoch` to adjust `noise_snr_range` dynamically.
    - Connect `Trainer` epoch loop to `audio_processor.augmentation`.
- [x] Task: Add Performance Toggles to UI and Backend f6171eb
    - Update `WakewordConfig` to include `use_compile` and `use_gradient_checkpointing`.
    - Update `Trainer.__init__` to apply `torch.compile` and `model.gradient_checkpointing_enable()` based on config.
    - Add checkboxes to `src/ui/panel_training.py`.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Training Loop Enhancements' (Protocol in workflow.md)

## Phase 2: Dual-Teacher & Distillation UI
Refine the Knowledge Distillation setup for expert users.

- [x] Task: Extend Configuration Panel for Secondary Teacher 078327f
- [x] Task: Implement Teacher Compatibility Validation 078327f
    - Add validation logic to `ConfigValidator` to check if teacher checkpoints exist and match architectures.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Dual-Teacher KD' (Protocol in workflow.md)

## Phase 3: Background Mining & Session Persistence
Enable long-form audio analysis for hard negative discovery.

- [x] Task: Implement Background Miner Engine 30328dd
- [x] Task: Integrate Background Miner into UI 30328dd
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Background Mining' (Protocol in workflow.md)

## Phase 4: Distributed Cascade Integration
Connect the Sentry (Edge) to the Judge (Server) for final verification.

- [x] Task: Implement Judge Client Logic fd3af7b
- [x] Task: Add Cascade Testing UI fd3af7b
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Distributed Cascade' (Protocol in workflow.md)
