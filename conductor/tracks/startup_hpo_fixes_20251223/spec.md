# Specification - Track: Startup and HPO Fixes

## Overview
This track addresses two critical bugs preventing the application from being usable:
1.  **Launch Failure:** `run.py` fails shortly after initialization (specifically after the Evaluation Panel is initialized on CUDA) due to a `NameError: name 'results' is not defined`.
2.  **HPO Failure:** Hyperparameter Optimization trials crash because `Trial.report` is incompatible with the current configuration, and a `best_f1` variable is referenced before assignment.

## Functional Requirements

### 1. Application Startup (`run.py` / `src/ui/app.py`)
-   Locate where `results` is referenced in the startup sequence (likely within `run.py` or the main UI initialization logic).
-   Fix the `NameError` by ensuring variables are properly defined or scoped before the application attempts to launch the server.
-   Ensure the application launches correctly to the Gradio UI and remains stable for interaction.

### 2. Hyperparameter Optimization (`src/training/hpo.py`)
-   **Initialize Variables:** Fix `UnboundLocalError` by ensuring `best_f1` is properly initialized before the checkpoint check logic.
-   **Optuna Compatibility:** Resolve the `NotImplementedError` regarding `Trial.report`. Since the user intends for Single Objective (F1 score) optimization, ensure the HPO logic correctly identifies the study type and only uses reporting/pruning features compatible with single-objective optimization.

## Non-Functional Requirements
-   **Stability:** The UI must be reachable and responsive after the fix.
-   **Logging:** Ensure initialization errors provide clear tracebacks in the console for faster debugging.

## Acceptance Criteria
-   [ ] `python run.py` successfully reaches the state where the Gradio interface is live and interactive.
-   [ ] Starting an HPO trial from the UI completes at least one full trial without crashing.
-   [ ] `best_f1` is tracked correctly across HPO trials without scoping errors.

## Out of Scope
-   Adding new HPO features or additional optimization objectives.
-   Fixing unrelated UI display issues unless they prevent the fix from being verified.
