# Implementation Plan: HPO Testing Module Optimization & Variable Expansion

This plan outlines the steps to optimize the HPO module for speed and expand the set of configurable variables in the HPO outputs and UI.

## Phase 1: Performance Analysis & Infrastructure Setup
Focus on understanding current bottlenecks and setting up the testing environment.

- [x] Task: Analyze current HPO implementation in `src/training/` and `src/ui/` to identify trial execution bottlenecks.
- [x] Task: Benchmark current HPO sweep time for a standard configuration.
- [x] Task: Identify all potential variables for inclusion in HPO output across architecture, training, data, and inference modules.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Performance Analysis & Infrastructure Setup' (Protocol in workflow.md)

## Phase 2: Speed Optimization Implementation
Implement hardware-aware parallelization and intelligent trial pruning.

- [ ] Task: Implement parallel trial execution using Optuna's `n_jobs` or distributed study capabilities, ensuring thread-safety.
- [ ] Task: Integrate Optuna Pruners (e.g., `MedianPruner`, `HyperbandPruner`) to implement early stopping for unpromising trials.
- [ ] Task: Add configuration options to the HPO settings UI to control parallelization and pruning aggressiveness.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Speed Optimization Implementation' (Protocol in workflow.md)

## Phase 3: Variable Expansion & Output Structuring
Update the HPO engine to track and return a comprehensive set of parameters.

- [ ] Task: Modify the HPO objective function to log and return all architectural, training, preprocessing, and threshold variables.
- [ ] Task: Standardize the HPO result data structure to support a wide range of variable types (int, float, categorical).
- [ ] Task: Update the HPO output persistence logic to ensure all variables are saved to the temporary results storage.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Variable Expansion & Output Structuring' (Protocol in workflow.md)

## Phase 4: UI Enhancement & Profile Integration
Create a rich, editable results view and robust "Apply/Save" functionality.

- [ ] Task: Design and implement a structured Gradio table/form to display HPO results with inline editing capabilities.
- [ ] Task: Implement logic to map UI-edited HPO results back to the internal configuration format.
- [ ] Task: Enhance the "Apply to Current Session" and "Save to Profile" buttons to handle the full expanded set of variables.
- [ ] Task: Verify that edited variables are correctly reflected in the configuration and active training session.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: UI Enhancement & Profile Integration' (Protocol in workflow.md)

## Phase 5: Final Verification & Benchmarking
Ensure performance gains and full variable coverage.

- [ ] Task: Run a final benchmark to quantify speed improvements from parallelization and pruning.
- [ ] Task: Perform an end-to-end test of the HPO-to-Profile flow, verifying that every single variable is correctly persisted.
- [ ] Task: Conductor - User Manual Verification 'Phase 5: Final Verification & Benchmarking' (Protocol in workflow.md)
