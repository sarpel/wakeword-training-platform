# Implementation Plan: Model Excellence & Optimization Suite

This plan follows a TDD-driven approach to implement advanced optimization techniques, dual-teacher distillation, and hard-negative mining for the wakeword platform.

## Phase 1: Advanced Metrics & Multi-Objective HPO [checkpoint: f64cb06]
**Goal:** Implement pAUC as a primary metric and configure the sequential multi-objective HPO engine.

- [x] Task: Implementation of the Partial AUC (pAUC) metric calculator in `src/evaluation/metrics.py`. e4f1073
- [x] Task: Update the training loop to report pAUC and Latency to the HPO tracker. d6ef1f4
- [x] Task: Configure Optuna with `MOTPE` or `NSGA-II` for sequential multi-objective optimization (Accuracy vs. Latency). d6ef1f4
- [x] Task: Implement the "Exploit-and-Explore" mutation logic for HPO trial hyperparameter forks. d6ef1f4
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Metrics & HPO' (Protocol in workflow.md)

## Phase 2: Dual-Teacher Knowledge Distillation [checkpoint: 8af3b1d]
**Goal:** Integrate a second teacher model and implement intermediate feature matching.

- [x] Task: Implement the secondary teacher architecture (Conformer/CNN-Transformer) and its loading logic. 8af3b1d
- [x] Task: Create the `DualTeacherDistiller` wrapper to aggregate signals from Wav2Vec2 and the new teacher. 8af3b1d
- [x] Task: Implement Feature Map Alignment loss (e.g., MSE on projector-aligned student/teacher features). 8af3b1d
- [x] Task: Implement dynamic temperature scaling logic (scheduler) for the distillation loss. 8af3b1d
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Dual-Teacher Distillation' (Protocol in workflow.md)

## Phase 3: Interactive Hard Negative Mining [checkpoint: 99bbef4]
**Goal:** Build the UI and backend loop for mining and verifying false positives.

- [x] Task: Add a "Mining" trigger to the Benchmark results logic to capture false positives. 99bbef4
- [x] Task: Implement the "Verification Queue" backend (JSON/SQLite) to store mined samples for review. 99bbef4
- [x] Task: Update the Gradio UI with a verification interface for "Confirm/Discard" actions. 99bbef4
- [x] Task: Implement the data injection logic to include verified negatives in the next training run. 99bbef4
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Hard Negative Mining' (Protocol in workflow.md)

## Phase 4: Pipeline Refinement & Final Integration [checkpoint: da213a9]
**Goal:** Enhance augmentations and perform final validation of the "Model Excellence" suite.

- [x] Task: Implement refined SpecAugment parameters in the `AudioAugmentor`. da213a9
- [x] Task: Implement the advanced noise mixing strategy with SNR-based scheduling. da213a9
- [x] Task: Run a full "theoretical best" training trial using all new components. da213a9
- [x] Task: Conductor - User Manual Verification 'Phase 4: Final Integration' (Protocol in workflow.md)
