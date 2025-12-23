# Specification: Dynamic CMVN Dimensions & Hardcoding Removal

## Overview
This track addresses the issue where the Cepstral Mean Variance Normalization (CMVN) statistics are effectively hardcoded to 40 mel bands, causing mismatches and forced disabling when users attempt to train with other configurations (e.g., 64 mels). The goal is to eliminate hardcoded "40" values across the pipeline and implement a user-aware mismatch handling system.

## Functional Requirements
- **Dynamic Feature Extraction:** Remove default `n_mels=40` from `FeatureExtractor` and ensure it always respects the active `WakewordConfig`.
- **Adaptive Model Architectures:** Update model builders to dynamically calculate input dimensions based on the feature configuration (n_mels/n_mfcc and context window).
- **UI Consistency:** Update all UI panels (Panel 1: Dataset, Panel 2: Config, Panel 3: Training) to use 64 mel bands as the standard default, consistent with `DataConfig`.
- **User-Triggered CMVN Recomputation:** 
    - Implement detection logic for CMVN stats dimension mismatches.
    - Display a clear warning in the UI when a mismatch is detected.
    - Provide an explicit "Update CMVN Stats" button to resolve the mismatch.
- **Soft Warning Workflow:** If training is started with a mismatch, display a prominent warning informing the user that CMVN will be disabled for the current run unless recomputed.

## Non-Functional Requirements
- **UI Responsiveness:** Mismatch detection should occur immediately upon configuration change or checkpoint selection.
- **Maintainability:** Replace all magic numbers related to feature dimensions with references to centralized configuration constants.

## Acceptance Criteria
- [ ] Training with any valid `n_mels` value (e.g., 64, 80, 128) no longer triggers a hardcoded dimension error.
- [ ] Changing `n_mels` in the UI correctly prompts the user to recompute CMVN stats.
- [ ] Models are successfully initialized with correct input shapes regardless of the feature configuration.
- [ ] The "40 Mel" hardcoding is verified as removed from `src/data/feature_extraction.py` and model architecture files.

## Out of Scope
- Automatic migration of old `cmvn_stats.json` files (recomputation is required).
- Changing the underlying math of CMVN normalization.
