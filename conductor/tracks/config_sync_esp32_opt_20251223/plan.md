# Implementation Plan: Configuration Synchronization and ESP32-S3 Optimization

## Phase 1: Configuration Synchronization & UI Validation
Goal: Implement logic to detect and handle mismatches between dataset features and active configuration.

- [x] Task: Create validation utility for configuration vs. dataset metadata. [9bbe256]
- [ ] Task: Implement the "Mismatch Prompt" logic in the Gradio UI.
- [ ] Task: Create unit tests for configuration validation (success and mismatch cases).
- [ ] Task: Task: Conductor - User Manual Verification 'Phase 1: Configuration Synchronization' (Protocol in workflow.md)

## Phase 2: `tiny_conv` Profile Refinement
Goal: Update the `tiny_conv` preset with optimized parameters for ESP32-S3.

- [ ] Task: Calculate and update `tiny_conv` preset in `src/config/presets.py` (Filters, Layers, Mel Bins).
- [ ] Task: Enable high-quality training components (Distillation, RIR) in the `tiny_conv` default configuration.
- [ ] Task: Verify that the updated model architecture compiles and fits within theoretical ESP32-S3 RAM limits.
- [ ] Task: Write unit tests to ensure the `tiny_conv` preset loads with the expected high-quality parameters.
- [ ] Task: Task: Conductor - User Manual Verification 'Phase 2: Profile Refinement' (Protocol in workflow.md)

## Phase 3: Export Path & ESPHome Integration
Goal: Automate model export to a fixed path for ESPHome compatibility.

- [ ] Task: Modify the export module to support a fixed path specifically for ESPHome Atom Echo.
- [ ] Task: Update the UI to reflect the new automated export location.
- [ ] Task: Create integration tests to verify the export file appears in the correct fixed directory.
- [ ] Task: Task: Conductor - User Manual Verification 'Phase 3: Export Path' (Protocol in workflow.md)
