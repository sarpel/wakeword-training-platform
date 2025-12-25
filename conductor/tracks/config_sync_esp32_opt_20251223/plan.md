# Implementation Plan: Configuration Synchronization and ESP32-S3 Optimization

## Phase 1: Configuration Synchronization & UI Validation [checkpoint: 7fa3b95]
Goal: Implement logic to detect and handle mismatches between dataset features and active configuration.

- [x] Task: Create validation utility for configuration vs. dataset metadata. [9bbe256]
- [x] Task: Implement the "Mismatch Prompt" logic in the Gradio UI. [6b86087]
- [x] Task: Create unit tests for configuration validation (success and mismatch cases). [9bbe256]
- [x] Task: Task: Conductor - User Manual Verification 'Phase 1: Configuration Synchronization' (Protocol in workflow.md) [7fa3b95]

## Phase 2: `tiny_conv` Profile Refinement [checkpoint: fa33570]
Goal: Update the `tiny_conv` preset with optimized parameters for ESP32-S3.

- [x] Task: Calculate and update `tiny_conv` preset in `src/config/presets.py` (Filters, Layers, Mel Bins). [8bc535c]
- [x] Task: Enable high-quality training components (Distillation, RIR) in the `tiny_conv` default configuration. [8bc535c]
- [x] Task: Verify that the updated model architecture compiles and fits within theoretical ESP32-S3 RAM limits. [8bc535c]
- [x] Task: Write unit tests to ensure the `tiny_conv` preset loads with the expected high-quality parameters. [8bc535c]
- [x] Task: Task: Conductor - User Manual Verification 'Phase 2: Profile Refinement' (Protocol in workflow.md) [fa33570]

## Phase 3: Export Path & ESPHome Integration [checkpoint: 5c61b24]
Goal: Automate model export to a fixed path for ESPHome compatibility.

- [x] Task: Modify the export module to support a fixed path specifically for ESPHome Atom Echo. [8649cfa]
- [x] Task: Update the UI to reflect the new automated export location. [8649cfa]
- [x] Task: Create integration tests to verify the export file appears in the correct fixed directory. [8649cfa]
- [x] Task: Task: Conductor - User Manual Verification 'Phase 3: Export Path' (Protocol in workflow.md) [5c61b24]
