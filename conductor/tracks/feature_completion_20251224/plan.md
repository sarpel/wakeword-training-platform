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

- [ ] Task: Extend Configuration Panel for Secondary Teacher
    - Add `secondary_teacher_architecture` and `secondary_teacher_model_path` fields to `src/ui/panel_config.py`.
    - Create `tests/test_dual_teacher_config.py` to ensure config values propagate to `DistillationTrainer`.
- [ ] Task: Implement Teacher Compatibility Validation
    - Add validation logic to `ConfigValidator` to check if teacher checkpoints exist and match architectures.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Dual-Teacher KD' (Protocol in workflow.md)

## Phase 3: Background Mining & Session Persistence
Enable long-form audio analysis for hard negative discovery.

- [ ] Task: Implement Background Miner Engine
    - Create `src/evaluation/background_miner.py` with session persistence (JSON).
    - Write tests in `tests/test_background_mining.py` for pause/resume logic.
- [ ] Task: Integrate Background Miner into UI
    - Add "Background Miner" sub-tab to the Mining Queue tab in `src/ui/panel_evaluation.py`.
    - Implement file browser for selecting background recordings.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Background Mining' (Protocol in workflow.md)

## Phase 4: Distributed Cascade Integration
Connect the Sentry (Edge) to the Judge (Server) for final verification.

- [ ] Task: Implement Judge Client Logic
    - Create `src/evaluation/judge_client.py` to handle HTTP POST requests to the Judge server.
    - Write tests with a mocked Judge server to verify response handling.
- [ ] Task: Add Cascade Testing UI
    - Add "Cascade Testing" sub-tab to Panel 4.
    - Add fields for Judge URL and a "Verify with Judge" action for microphone/file results.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Distributed Cascade' (Protocol in workflow.md)
