# Plan: Visual Analysis & Debugging Suite

## Phase 1: Backend Analysis Engine
*Goal: Create the logic for threshold analysis and data collection.*

- [x] Task: Implement `ThresholdAnalyzer` in `src/evaluation/advanced_evaluator.py` to compute metrics for varying thresholds (f1da0ae)
    - [ ] Write Tests
    - [ ] Implement Feature
- [ ] Task: Create a utility to collect and serialize False Positive samples for UI consumption
    - [ ] Write Tests
    - [ ] Implement Feature
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Backend Analysis Engine' (Protocol in workflow.md)

## Phase 2: Interactive UI Components
*Goal: Build the Gradio interface for debugging and tuning.*

- [ ] Task: Extend `src/ui/panel_evaluation.py` with an "Analysis Dashboard" section
    - [ ] Write Tests
    - [ ] Implement Feature
- [ ] Task: Implement the "False Positive Inspector" gallery with audio playback
    - [ ] Write Tests
    - [ ] Implement Feature
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Interactive UI Components' (Protocol in workflow.md)

## Phase 3: Observability & Integration
*Goal: Integrate benchmarking metrics and polish the experience.*

- [ ] Task: Update UI to display real-time latency and memory metrics during inference
    - [ ] Write Tests
    - [ ] Implement Feature
- [ ] Task: Final end-to-end validation of the debugging suite
    - [ ] Write Tests
    - [ ] Implement Feature
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Observability & Integration' (Protocol in workflow.md)
