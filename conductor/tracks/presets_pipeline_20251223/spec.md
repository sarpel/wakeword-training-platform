# Track Specification: Presets & Pipeline Integration (CMVN, Thresholds, Streaming, & Size Targets)

## Overview
This track focuses on enhancing the project's configuration system by integrating missing professional-grade parameters into `src/config/presets.py` and ensuring the training/export pipeline actively utilizes these settings. The goal is to move from a basic configuration to an integrated, production-ready pipeline that handles normalization, streaming stability, and hardware constraints.

## Functional Requirements
1.  **Enhanced Presets:**
    *   Add **CMVN (Cepstral Mean/Variance Normalization)** configuration to global defaults.
    *   Define **Detection Threshold Targets** (FAH, EER goals) within training profiles.
    *   Introduce **Streaming Detection Parameters**:
        *   Hysteresis thresholds (Upper/Lower).
        *   Buffer lengths (ms).
        *   Smoothing windows (frame counts).
    *   Define **Model Size Targets** (Flash/RAM) for hardware-specific profiles (e.g., ESP32, Raspberry Pi).
    *   Configure **Quantization-Aware Calibration** settings, including a representative mix ratio for calibration data.

2.  **Pipeline Integration:**
    *   **CMVN Implementation:** Ensure the training and inference engines actively apply CMVN using the new preset defaults.
    *   **Streaming Stability:** Update the streaming detection logic to respect the hysteresis and smoothing parameters from the active profile.
    *   **Calibration Logic:** Implement the representative mix sampling (positive vs. negative ratio) for quantization-aware training (QAT) and post-training quantization (PTQ).

3.  **Hardware Validation:**
    *   Implement a post-export check that compares final model size (TFLite/ONNX) against the targets defined in the preset.
    *   If targets are exceeded, the system must trigger a high-visibility warning in the logs and UI dashboard.

## Non-Functional Requirements
*   **Performance:** CMVN application should not introduce significant latency to the feature extraction pipeline.
*   **Observability:** All validation warnings regarding model size or calibration failures must be logged using the existing structured logging (`structlog`).

## Acceptance Criteria
- [ ] `presets.py` contains all new parameters for at least the 'Standard' and 'Lightweight' profiles.
- [ ] Training logs confirm CMVN is being applied to features.
- [ ] A test script verifies that the streaming logic correctly implements hysteresis based on preset values.
- [ ] Exporting a model that exceeds defined RAM/Flash limits produces a clear "Warning" status without halting the process.
- [ ] Quantization calibration utilizes the specified representative mix of positive and negative samples.

## Out of Scope
*   Automatic model pruning or architectural changes to force-fit models into memory (Adaptive Optimization).
*   Implementation of per-profile custom CMVN logic (staying with Global Defaults for now).
