# Track Specification: Training & Performance Improvements (Focal Loss & QAT Accuracy)

## Overview
This track focuses on enhancing the core training pipeline of the Wakeword Training Platform. By implementing Focal Loss, we aim to improve the model's ability to distinguish between the wakeword and difficult background noises (hard negatives). Simultaneously, we will implement strategies within the Quantization-Aware Training (QAT) workflow to ensure that the transition from high-precision training to low-precision (INT8) edge deployment maintains maximum accuracy.

## Functional Requirements
- **Focal Loss Implementation:**
    - Integrate `FocalLoss` as a configurable option in the training pipeline.
    - Provide tunable parameters for `gamma` (focusing parameter) and `alpha` (weighting factor).
    - Update the trainer to handle weighted loss calculations during the optimization step.
- **QAT Accuracy Recovery:**
    - Implement a multi-stage training routine: Standard Training -> QAT Fine-tuning.
    - Integrate support for collecting activation statistics (calibration) prior to quantization.
    - Provide a "Quantization Error" report comparing FP32 vs. INT8 performance on the validation set.

## Non-Functional Requirements
- **Performance:** Focal Loss implementation should not significantly increase training time per epoch (<5% overhead).
- **Stability:** The QAT pipeline must reliably export to TFLite/ONNX without breaking existing conversion logic.
- **Maintainability:** Ensure loss functions and QAT logic are modular and well-tested.

## Acceptance Criteria
- [ ] Training logs confirm higher relative weighting for high-loss samples when Focal Loss is enabled.
- [ ] The model successfully undergoes QAT and exports to a functional INT8 TFLite model.
- [ ] Accuracy drop after INT8 quantization is measured and is less than 2% relative to the FP32 baseline on standard benchmarks.
- [ ] Unit tests verify the mathematical correctness of the Focal Loss implementation.

## Out of Scope
- Architectural changes to the base models (MobileNetV3/Conformer).
- Hyperparameter Optimization (HPO) for the new Focal Loss parameters (this can be a separate track).
