# Implementation Plan: Advanced Optimization for TinyConv

This plan follows the Test-Driven Development (TDD) methodology and phase checkpointing protocol defined in `conductor/workflow.md`.

## Phase 1: Architecture Upgrade (TinyConv V2) [checkpoint: f053a1e]
- [x] Task: Implement Depthwise Separable Convolution architecture c01c550
    - [ ] Create `tests/test_tiny_conv_v2.py` with failing tests for parameter count verification and forward pass.
    - [ ] Refactor `TinyConvWakeword` in `src/models/architectures.py` to support Depthwise Separable blocks via a `use_depthwise` flag.
    - [ ] Verify parameter reduction targets (~70% reduction) using the test suite.
- [x] Task: Update UI and Configuration for TinyConv V2 73a09f7
    - [ ] Update `ModelConfig` in `src/config/defaults.py` to include `tiny_conv_use_depthwise`.
    - [ ] Add the toggle to the Gradio UI in `src/ui/panel_config.py`.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Architecture Upgrade' (Protocol in workflow.md)

## Phase 2: QAT Stabilization (Module Fusion) [checkpoint: 4e36e28]
- [x] Task: Implement Automated Fusion Engine bfdb9bd
    - [ ] Create `tests/test_qat_fusion.py` to verify that `Conv+BN+ReLU` layers are correctly collapsed into a single fused module.
    - [ ] Implement `fuse_tiny_conv` logic in `src/training/qat_utils.py` using `torch.ao.quantization.fuse_modules`.
    - [ ] Update `prepare_model_for_qat` to automatically invoke fusion for supported architectures.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: QAT Stabilization' (Protocol in workflow.md)

## Phase 3: Advanced Dual-Teacher Distillation Upgrades [checkpoint: 0c3dcec]
- [x] Task: Implement Learnable Projectors d3a020b
    - [ ] Create `tests/test_distillation_projectors.py` to verify distillation between mismatched dimensions (e.g., 768 -> 64).
    - [ ] Implement `Projector` class and dynamic injection logic in `src/training/distillation_trainer.py`.
- [x] Task: Implement Soft-Confidence (Dynamic) Weighting fb78b69
    - [ ] Create tests to verify that loss weights shift dynamically based on teacher entropy.
    - [ ] Update `compute_loss` in `DistillationTrainer` to calculate and apply dynamic teacher weights.
- [x] Task: Expert Layer Selection Logic 91682aa
    - [ ] Update `DistillationConfig` to allow a list of `teacher_alignment_layers`.
    - [ ] Modify `DistillationTrainer` to hook into specific internal layers of Wav2Vec2 and Conformer based on config.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Advanced Dual-Teacher Distillation' (Protocol in workflow.md)

## Phase 4: Documentation & Final QA
- [x] Task: Expert Distillation Guide 1259d60
    - [ ] Create `docs/EXPERT_DISTILLATION_GUIDE.md` with detailed explanations of layer selection and trade-offs.
- [ ] Task: Final E2E Integration Benchmark
    - [ ] Run a full training run using TinyConvV2 + Dual Distillation + QAT Fusion and verify the INT8 accuracy drop.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Documentation & Final QA' (Protocol in workflow.md)
