# Specification: Configuration Synchronization and ESP32-S3 Optimization

## Overview
This track aims to eliminate potential shape mismatches in the wakeword training pipeline by synchronizing dataset feature extraction with the active configuration (presets or saved configs). Additionally, it involves updating the `tiny_conv` profile to maximize model quality while staying within the ESP32-S3's resource constraints and ensuring compatibility with ESPHome Atom Echo firmware.

## Functional Requirements
- **Config-Dataset Synchronization:**
    - Detect mismatches between the active configuration (sample rate, mel bins, window size, etc.) and existing extracted features.
    - Implement a UI prompt in the Gradio interface to alert the user of mismatches and offer options to re-extract or revert.
- **`tiny_conv` Profile Optimization:**
    - Update the `tiny_conv` preset based on new calculations for fitting probability into the ESP32-S3 MCU.
    - Prioritize and increase: Input Resolution (Mel bins), Model Capacity (filters/layers), Advanced Augmentation (RIR, background noise), and Knowledge Distillation.
- **ESPHome Compatibility:**
    - Configure a fixed export path for trained models to ensure they are saved in a location compatible with the ESPHome Atom Echo firmware build environment.

## Non-Functional Requirements
- **Resource Constraints:** The updated `tiny_conv` model MUST fit within the ESP32-S3 RAM/Flash limits as defined by the new calculations.
- **UI Integrity:** Prevent "project lifetime" shape mismatches by enforcing consistency between data and model configuration.

## Acceptance Criteria
- Loading a config/preset triggers a mismatch warning if existing features are incompatible.
- The `tiny_conv` profile includes the "high quality" training components (distillation, augmentation, etc.) by default.
- Models exported using the `tiny_conv` profile successfully fit on the ESP32-S3 target.
- Exported models appear in the designated fixed path for ESPHome.

## Out of Scope
- Implementing the ESPHome firmware itself (assumed to be already available/flashed).
- Modifying teacher models (only their application in distillation is included).
