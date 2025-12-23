# Implementation Plan - Training & Performance Improvements (Focal Loss & QAT Accuracy)

## Phase 1: Focal Loss Implementation
- [x] Task: Write unit tests for Focal Loss implementation (verify gradients and focusing logic)
- [x] Task: Implement `FocalLoss` class in `src/models/losses.py` (or equivalent loss module)
- [x] Task: Update the training configuration schema and Trainer to support `FocalLoss` as a criterion
- [x] Task: Conductor - User Manual Verification 'Phase 1: Focal Loss Implementation' (Protocol in workflow.md)

## Phase 2: QAT Accuracy Recovery Pipeline
- [x] Task: Write tests for the QAT fine-tuning flow (verify state dict loading and quantization prep)
- [x] Task: Implement QAT fine-tuning logic in the training pipeline
- [x] Task: Create a calibration utility to collect activation statistics before INT8 export
- [x] Task: Implement the "Quantization Error" reporting tool to compare FP32 vs. INT8 validation metrics
- [x] Task: Conductor - User Manual Verification 'Phase 2: QAT Accuracy Recovery' (Protocol in workflow.md)

## Phase 3: Integration and Benchmarking
- [~] Task: Write integration tests for the full pipeline (Standard Training -> QAT -> TFLite Export)
- [ ] Task: Perform a benchmark run to verify that accuracy drop is < 2% and Focal Loss improves hard-negative handling
- [ ] Task: Final code cleanup, documentation update, and coverage verification (>80%)
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Integration and Benchmarking' (Protocol in workflow.md)
