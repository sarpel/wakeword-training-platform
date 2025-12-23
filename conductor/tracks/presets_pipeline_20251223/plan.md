# Implementation Plan: Presets & Pipeline Integration

This plan outlines the steps to integrate professional-grade parameters (CMVN, Streaming Stability, Size Targets, and Calibration) into the project's configuration and training pipeline.

## Phase 1: Configuration & Validation Schema (Red/Green/Refactor) [checkpoint: e35a807]
Goal: Define the new parameters in the configuration system and ensure they are validated.

- [x] **Task: Define New Schema in `src/config/validator.py` or `pydantic_validator.py`** e35a807
    - Add Pydantic models for `CMVNConfig`, `StreamingConfig`, `SizeTargetConfig`, and `CalibrationConfig`.
- [x] **Task: Update `presets.py` with New Parameters** e35a807
    - Inject default values for CMVN, Streaming (Hysteresis, Buffers, Smoothing), Size Targets, and Calibration Mix into existing profiles.
- [x] **Task: Verify Configuration Loading** e35a807
    - Write tests to ensure the new parameters are correctly loaded and validated when a profile is selected.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Configuration & Validation Schema' (Protocol in workflow.md) e35a807

## Phase 2: Feature Normalization & Calibration Logic [checkpoint: ef1c147]
Goal: Implement the underlying logic for CMVN and balanced calibration sampling.

- [x] **Task: Implement/Integrate CMVN in Feature Pipeline** ef1c147
    - Update `src/data/feature_extraction.py` or `processor.py` to apply CMVN using the preset configuration.
- [x] **Task: Implement Balanced Calibration Sampler** ef1c147
    - Update `src/data/balanced_sampler.py` or create a utility to select the "Representative Mix" for quantization calibration.
- [x] **Task: TDD - Verification of Feature Scaling** ef1c147
    - Write tests to ensure audio features are correctly normalized when CMVN is enabled.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Feature Normalization & Calibration Logic' (Protocol in workflow.md) ef1c147

## Phase 3: Streaming Stability & Post-Export Validation
Goal: Implement the logic for stable streaming and hardware limit warnings.

- [ ] **Task: Update Streaming Inference Logic**
    - Modify the streaming engine (e.g., `src/data/processor.py` or a dedicated streaming class) to implement hysteresis and temporal smoothing.
- [ ] **Task: Implement Model Size Post-Check**
    - Add a check in the export pipeline (e.g., `src/export/`) to compare the exported file size against `SizeTargetConfig`.
- [ ] **Task: TDD - Verify Streaming Hysteresis**
    - Create a test case with oscillating confidence scores to ensure the "Wake" state remains stable according to preset thresholds.
- [ ] **Task: TDD - Verify Size Warning Trigger**
    - Mock an export that exceeds targets and verify that a warning is logged without halting the process.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Streaming Stability & Post-Export Validation' (Protocol in workflow.md)

## Phase 4: UI/Dashboard Integration
Goal: Expose these new metrics and warnings to the user.

- [ ] **Task: Update UI to Display Size Warnings**
    - Modify the Gradio dashboard to show a warning badge if model size targets are exceeded.
- [ ] **Task: Log Calibration Metrics**
    - Ensure quantization calibration statistics are visible in logs/Weights & Biases.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: UI/Dashboard Integration' (Protocol in workflow.md)
