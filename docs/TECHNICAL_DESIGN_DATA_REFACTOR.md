# Technical Design Specification: Dataset & Feature Extraction Refactoring

## 1. Objective
Enhance the robustness, flexibility, and documentation of the data pipeline by addressing hardcoded constraints and implicit behaviors.

## 2. Scope
- `src/data/dataset.py`: Refactor label mapping and fallback logic.
- `src/data/feature_extraction.py`: Correction of docstrings.
- `src/config/defaults.py`: (Implicit) Add necessary config fields if they don't exist.

## 3. Component Design

### 3.1 Dynamic Label Mapping (`src/data/dataset.py`)

**Current State:**
Hardcoded dictionary in `_create_label_map`:
```python
label_map = {
    'positive': 1,
    'negative': 0,
    'hard_negative': 0
}
```

**Proposed Design:**
Inject label mapping via constructor. If not provided, default to a binary classification schema but allow for extensibility.

**Implementation Plan:**
1.  Update `WakewordDataset.__init__` to accept `class_mapping: Optional[Dict[str, int]] = None`.
2.  In `_create_label_map`, if `class_mapping` is provided, use it.
3.  Otherwise, use the default binary map (keeping backward compatibility).
4.  Add validation: Ensure all categories found in `manifest['files']` exist in the map.

### 3.2 Explicit Fallback Logic (`src/data/dataset.py`)

**Current State:**
In `__getitem__`:
```python
if self.use_precomputed_features_for_training:
    features = self._load_from_npy(...)
    if features is not None:
        return ...
    elif not self.fallback_to_audio:
        logger.warning("...")
        # Falls through to audio loading regardless of flag!
```

**Proposed Design:**
Strictly enforce `fallback_to_audio` flag.

**Implementation Plan:**
1.  Modify `__getitem__` logic:
    ```python
    if self.use_precomputed_features_for_training:
        features = self._load_from_npy(file_info, idx)
        if features is not None:
             # ... return NPY features ...
        
        # NPY missing or failed
        if not self.fallback_to_audio:
            raise FileNotFoundError(f"NPY features missing for {file_path} and fallback_to_audio=False")
    ```
2.  Update `load_dataset_splits` to propagate these flags correctly.

### 3.3 Documentation Correction (`src/data/feature_extraction.py`)

**Current State:**
Docstrings imply GPU/Device agnostic behavior, but code enforces `cpu`.

**Proposed Design:**
Explicitly document that this class is designed for CPU-based preprocessing within `DataLoader` workers.

**Implementation Plan:**
1.  Update class docstring.
2.  Update `__init__` docstring to explain `device` parameter is forced/ignored.

## 4. Action Items

1.  **Edit `src/data/dataset.py`**:
    -   Add `class_mapping` argument to `__init__`.
    -   Refactor `_create_label_map` to use injected mapping.
    -   Refactor `__getitem__` to raise Error if NPY is missing and fallback is False.
    -   Update `load_dataset_splits` to accept `class_mapping`.

2.  **Edit `src/data/feature_extraction.py`**:
    -   Update docstrings to reflect CPU-only design.

3.  **Validation**:
    -   Verify `dataset.py` can still load default binary data.
    -   Verify `dataset.py` throws error when `fallback_to_audio=False` and NPY is missing.

## 5. Risk Assessment
-   **Breaking Change**: The strict fallback logic might break existing training pipelines if they implicitly relied on the buggy fallback behavior.
    -   *Mitigation*: Ensure default config has `fallback_to_audio=True` if that is the desired safe default, or communicate this change clearly. (For this task, we will implement the strict check as requested for correctness).
