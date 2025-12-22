# Plan: Evaluation and Export Fixes

## Phase 1: Research and Error Reproduction
- [x] Task: Audit class mapping and label handling across `src/training`, `src/evaluation`, and `src/ui`.
- [x] Task: Identify specific "operator not supported" errors by attempting a TFLite export with a QAT model.
- [x] Task: Review preprocessing transformations in `src/data/preprocessing.py` and `src/data/processor.py` for training vs. evaluation discrepancies.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Research and Error Reproduction' (Protocol in workflow.md)

## Phase 2: Fix Evaluation Logic and Metrics (TDD)
- [~] Task: Create a reproduction test case in `tests/test_evaluator_alignment.py` that demonstrates the FNR/F1 discrepancy using a mock "perfect" model.
- [x] Task: Standardize class label indices (Positive=1, Negative=0) across all modules.
- [x] Task: Update `src/evaluation/evaluator.py` and `src/evaluation/metrics.py` to ensure correct FNR and F1 calculation.
- [x] Task: Implement logging in the evaluation pipeline to report class mapping and label distribution for debugging.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Fix Evaluation Logic and Metrics' (Protocol in workflow.md)

## Phase 3: Repair ONNX and TFLite Export Pipeline (TDD)
- [x] Task: Create unit tests in `tests/test_tflite_export.py` for standard, QAT, and Quantized TFLite exports.
- [x] Task: Update `src/export/onnx_exporter.py` (and associated scripts) to resolve TFLite conversion fidelity issues.
- [x] Task: Implement robust QAT-to-TFLite export logic, ensuring feature consistency and quantization parameter retention.
- [x] Task: Add comprehensive error handling and descriptive logging to the ONNX export panel in `src/ui/panel_export.py`.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Repair ONNX and TFLite Export Pipeline' (Protocol in workflow.md)

## Phase 4: Verification and Finalization
- [x] Task: Run full test suite including new evaluation and export tests.
- [x] Task: Perform a benchmark comparison of ONNX vs TFLite inference performance.
- [x] Task: Quantify the F1 score restoration (verifying the delta between expected and observed results).
- [x] Task: Document findings, export limitations, and resolution steps in `DOCUMENTATION.md`.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Verification and Finalization' (Protocol in workflow.md)
