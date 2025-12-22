# Specification: Evaluation and Export Fixes

## 1. Overview
This track addresses two critical areas of the Wakeword Training Platform:
1.  **Evaluation Metric Discrepancy:** A bug where the evaluator reports >99% False Negative Rate (FNR) despite high training performance (F1 > 0.99). This is likely due to a positive/negative class label misalignment between the training process and the evaluation panel.
2.  **ONNX/TFLite Export Failures:** The ONNX export panel fails when converting to TFLite, particularly with Quantization-Aware Training (QAT) models, and encounters specific operator/conversion errors.

## 2. Functional Requirements

### 2.1 Evaluation & Metrics Logic
-   **Class Alignment:** Ensure the "positive" (wakeword) and "negative" (non-wakeword) class indices are consistent across `training/`, `evaluation/`, and `ui/`.
-   **Data Mapping:** Verify that feature vectors and labels align perfectly between training and evaluation datasets.
-   **Metric Validation:** Review and correct the implementation of FNR and F1 score calculations in `src/evaluation/`.
-   **Output Interpretation:** Confirm the model's raw output (logits/probabilities) is correctly parsed and thresholded during evaluation.
-   **Preprocessing Parity:** Ensure identical audio preprocessing/augmentation steps are applied during both training and evaluation.
-   **Class Imbalance:** Assess how class imbalance is handled in the evaluation suite and its impact on reported metrics.

### 2.2 ONNX to TFLite Export Pipeline
-   **Conversion Fidelity:** Fix the `onnx2tf` or TFLite conversion logic to prevent "operator not supported" errors.
-   **QAT Support:** Enable reliable TFLite export for models trained with Quantization-Aware Training.
-   **Quantized Output:** Validate that quantized TFLite exports (INT8/Float16) maintain performance parity.
-   **Feature Consistency:** Ensure specific model features (input/output names and shapes) are preserved during the conversion process.
-   **Error Handling:** Implement robust error catching and user-friendly logging in the export panel.

### 2.3 Verification & Documentation
-   **Unit Testing:** Implement unit tests for data loading and metric calculation modules.
-   **Integration Testing:** Add TFLite export-specific unit tests.
-   **Performance Benchmarking:** Benchmark TFLite inference performance to ensure no significant degradation.
-   **Logging:** Add detailed logging for data alignment checks and the ONNX export process.

## 3. Acceptance Criteria
-   Evaluator reports F1 scores and FNR that align with training performance (e.g., F1 > 0.99 for a "perfect" model).
-   The ONNX export panel successfully produces TFLite and Quantized TFLite models from both standard and QAT checkpoints.
-   All new unit tests for metrics and exports pass.
-   Export limitations and resolutions are documented in `DOCUMENTATION.md` or a specific export guide.

## 4. Out of Scope
-   Redesigning the core training loop architectures.
-   Adding new model types not currently supported by the platform.
