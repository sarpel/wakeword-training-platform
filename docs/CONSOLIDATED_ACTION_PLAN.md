# Consolidated Action Plan: Wakeword Project Remediation

This plan synthesizes critical fixes and architectural improvements for the Wakeword Project.

## Phase 0: Critical Runtime Fixes (Immediate Priority)
*Goal: Ensure the application runs without crashing due to missing imports or undefined variables.*
*Source: wakeword_project_analysis_report.md*

1.  **Fix Undefined Names & Missing Imports**:
    -   `src/export/onnx_exporter.py`: Fix `onnx`, `ort`, `np` undefined errors (16 errors reported).
    -   `src/evaluation/evaluator.py`: Import `time`, `enforce_cuda`, `AudioProcessor`, `FeatureExtractor`, `MetricsCalculator`. Implement/Import missing methods (`evaluate_file`, `evaluate_files`, `evaluate_dataset`, `get_roc_curve_data`, `evaluate_with_advanced_metrics`).
    -   `src/ui/panel_export.py`: Import `time`, `export_model_to_onnx`, `validate_onnx_model`, `benchmark_onnx_model`.
    -   `src/ui/panel_evaluation.py`: Import `time`, `SimulatedMicrophoneInference`, `WakewordDataset`, `MetricResults`.
    -   `src/training/checkpoint.py`: Fix `Trainer` type hint and `MetricResults` import.
    -   `src/training/checkpoint_manager.py`: Import `json`, `Trainer`, `shutil`.
    -   `src/data/dataset.py`: Fix `splits_dir` scope issue (line 549) - should be `data_root / 'splits'`.
    -   `src/config/logger.py`: Fix `get_logger` definition/call (should likely be `get_data_logger` or defined).
    -   `src/evaluation/dataset_evaluator.py`: Import `time`, `Path`.
    -   `src/evaluation/advanced_evaluator.py`: Define `calculate_comprehensive_metrics`.

## Phase 1: Architecture & Refactoring (Medium Priority)
*Goal: Improve maintainability, flexibility, and performance.*
*Source: TECHNICAL_DESIGN_DATA_REFACTOR.md, CODE_REVIEW_REPORT.md*

1.  **Dataset Improvements (Design Spec 3.1 & 3.2)**:
    -   **Dynamic Label Mapping**: Update `WakewordDataset.__init__` to accept a custom `class_mapping: Optional[Dict[str, int]]`. Refactor `_create_label_map` to use it.
    -   **Explicit Fallback Logic**: Refactor `__getitem__` to strictly respect `fallback_to_audio`. Raise `FileNotFoundError` if NPY is missing and fallback is `False`.
    -   **Propagation**: Update `load_dataset_splits` to accept and pass these new parameters.
2.  **Performance Optimization**:
    -   `src/data/splitter.py`: Implement O(1) NPY lookup using a pre-built index instead of `rglob` inside loops.
    -   `src/data/dataset.py`: Implement LRU cache for features (`@lru_cache`) instead of unbounded dictionary.
3.  **Code Cleanup**:
    -   **Line Endings**: configure `.gitattributes` to enforce LF for Python files.
    -   **Docstrings**: Update `src/data/feature_extraction.py` to explicitly state CPU-only design (Design Spec 3.3).