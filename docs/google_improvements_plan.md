# Google Improvements Plan for Wakeword Training Platform

## Introduction

This document outlines a comprehensive plan for improving the Wakeword Training Platform. The project is well-structured and includes many advanced features. The following recommendations are intended to further enhance its robustness, maintainability, and user experience.

## High-Level Recommendations

1.  **Configuration & Paths:** Refactor hardcoded paths and improve the clarity of configuration options.
2.  **Data Pipeline:** Enhance the data pipeline to provide a more comprehensive view of the dataset and improve the robustness of the `.npy` feature handling.
3.  **Training & Evaluation:** Refactor duplicated code and add more flexibility to the training and evaluation processes.
4.  **UI & User Experience:** Improve error handling and user guidance throughout the application.

## Detailed Improvement Plan

### 1. Configuration (`src/config`)

| Issue | Recommendation | Benefit | Priority |
| :--- | :--- | :--- | :--- |
| **Hardcoded Paths** | In `src/ui/panel_training.py`, paths like `data/raw/background` and `data/raw/rirs` are hardcoded. These should be derived from the `data_root` path provided in the configuration. | Improved flexibility and easier project restructuring. | Medium |
| **Ambiguous `use_precomputed_features`** | Rename the `use_precomputed_features` flag to `use_precomputed_features_for_training` to make its impact on the training process (skipping augmentation) more explicit. Add a warning in the UI when this is enabled. | Prevents unexpected behavior and improves user understanding. | High |

### 2. Data Processing (`src/data`)

| Issue | Recommendation | Benefit | Priority |
| :--- | :--- | :--- | :--- |
| **Incomplete Dataset Scan** | The `DatasetScanner` in `src/data/splitter.py` currently skips the `background` and `rirs` folders. Modify the scanner to include these folders in the scan. This will provide a more complete dataset health check. | A more accurate and comprehensive dataset health report. | Medium |
| **Robust `.npy` Path Finding** | The `_find_npy_path` function in `src/data/splitter.py` has a good fallback mechanism, but it can be made more robust by also searching for `.npy` files with slightly different naming conventions (e.g., with additional metadata in the filename). | Increased flexibility in handling `.npy` files from different sources. | Low |

### 3. Training (`src/training`)

| Issue | Recommendation | Benefit | Priority |
| :--- | :--- | :--- | :--- |
| **Duplicated Batch Processing Logic** | The `train_epoch` and `validate_epoch` methods in `src/training/trainer.py` contain similar code for iterating over batches. Refactor this into a shared `_run_epoch` method to reduce code duplication. | Improved maintainability and reduced code complexity. | Medium |
| **Hardcoded CUDA Device** | The training and evaluation pipelines enforce the use of CUDA. While this is good for performance, adding a CPU fallback option (with a clear warning about performance) would make the project more accessible for users without a GPU. | Increased accessibility and easier debugging on machines without a GPU. | Medium |

### 4. Evaluation (`src/evaluation`)

| Issue | Recommendation | Benefit | Priority |
| :--- | :--- | :--- | :--- |
| **Redundant `DataLoader` Creation** | The `evaluate_dataset` and `get_roc_curve_data` functions in `src/evaluation/evaluator.py` both create their own `DataLoader`. It would be more efficient to pass the existing `val_loader` or `test_loader` from the `Trainer` or create a single `DataLoader` for all evaluation tasks. | Improved efficiency and reduced resource consumption during evaluation. | Medium |

### 5. UI (`src/ui`)

| Issue | Recommendation | Benefit | Priority |
| :--- | :--- | :--- | :--- |
| **User-Friendly Error Messages** | The UI currently displays raw error messages. Implement a more user-friendly error reporting system that provides clear explanations and actionable suggestions for fixing the problem. | Improved user experience and easier troubleshooting for non-expert users. | High |
| **Centralized State Management** | The application uses a global `training_state` object. For a larger and more complex application, consider using a more structured state management solution (e.g., a dedicated state management library or a more structured global state object) to improve maintainability. | Improved code organization and easier state management as the application grows. | Low |
