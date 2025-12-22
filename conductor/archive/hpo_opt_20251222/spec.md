# Track Specification: HPO Testing Module Optimization & Variable Expansion

## Overview
This track focuses on enhancing the Hyperparameter Optimization (HPO) testing module within the training panel. The primary goals are to maximize testing speed through efficiency optimizations and to expose a comprehensive set of variables in the HPO outputs. This will ensure that "Apply" and "Save to Profile" actions capture all relevant configurations, allowing for highly detailed profile editing.

## Functional Requirements
- **Efficiency Optimization**:
    - Implement parallel execution of trials to leverage available hardware.
    - Implement early stopping (pruning) mechanisms to terminate unpromising trials quickly.
- **Variable Expansion**:
    - Expose all critical variables in the HPO output, including:
        - Model architecture parameters (layers, units, activations).
        - Training hyperparameters (learning rate, batch size, optimizer).
        - Data preprocessing/augmentation parameters.
        - Post-processing and inference thresholds.
- **UI Enhancements**:
    - Present HPO results in a structured table or form.
    - Allow users to edit these variables directly within the UI before applying or saving them to a profile.

## Non-Functional Requirements
- **Performance**: Minimize the time required to complete a full HPO sweep.
- **Usability**: Ensure the expanded variable list is organized and intuitive for the user to review and edit.

## Acceptance Criteria
- [ ] HPO trials can run in parallel (if hardware permits).
- [ ] Early stopping successfully prunes trials that fall below a performance threshold.
- [ ] The HPO result view displays an expanded list of parameters across architecture, training, data, and inference.
- [ ] Users can edit the HPO-discovered values in the UI.
- [ ] Clicking "Apply" or "Save to Profile" correctly persists the (possibly edited) expanded variable set.

## Out of Scope
- Redesigning the core HPO algorithm (e.g., switching from Optuna to a different engine).
- Real-time visualization of trial metrics (beyond what currently exists).
