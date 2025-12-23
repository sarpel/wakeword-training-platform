# Specification - Stability & UX Enhancement Suite (v1)

## Overview
This track addresses a cluster of stability issues, terminal output noise, and configuration management gaps identified during HPO and training sessions. The goal is to ensure reliable quantization reporting, clean logs, and better persistence of HPO-derived parameters.

## Functional Requirements

### 1. HPO Profile Management
- **Load Latest Button:** Implement a "Load Latest HPO Profile" button in the Gradio Training panel.
- **Persistence:** This button must read the most recently written HPO configuration from `configs/profiles/` (or the relevant disk location) and update the UI/session state.

### 2. Stability & Bug Fixes
- **QAT Report Channel Mismatch:** Fix the bug where `Generating Quantization Error Report` fails with a channel mismatch (e.g., `expected 1 channels, but got 32`). This occurs at the end of HPO trials and early stopping triggers.
- **Connection Error Suppression:** Suppress or gracefully catch `ConnectionResetError: [WinError 10054]` in the async connection logic to prevent stdout misalignment.

### 3. Preset Updates
- **`tiny_conv (esp32s3)` Defaults:** Update the existing preset with:
    - Default audio duration: 1.5 seconds.
    - TCN channels: 64.
    - Mel bands: 64.

## Non-Functional Requirements
- **Logging Clarity:** Ensure fixed logging does not introduce new duplicated outputs.
- **UX Consistency:** The "Load" functionality should provide immediate visual feedback in the UI when parameters are updated.

## Acceptance Criteria
- [ ] Clicking "Load Latest HPO Profile" successfully populates the training parameters with the latest disk-saved HPO results.
- [ ] HPO trials complete (including early stopping) without the "Failed to generate QAT report" channel mismatch error.
- [ ] The `ConnectionResetError` no longer clutters the terminal during active sessions.
- [ ] The `tiny_conv` preset loads with 1.5s/64/64 values by default.

## Out of Scope
- Investigative work on Seeding (Seed 42) and Seed mechanism enlightenment.
- Fixing duplicated terminal logs.
- Investigating `return_raw_audio` logic.
- Exclusive `alpha` vs `class_weights` logic.
