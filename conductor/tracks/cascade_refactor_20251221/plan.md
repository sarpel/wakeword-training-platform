# Plan: Refactor and Standardize Distributed Cascade Inference

## Phase 1: Foundation and Interfaces
*Goal: Define the abstract interfaces and project structure for modular inference.*

- [x] Task: Define `InferenceEngine` and `StageBase` abstract classes in `src/evaluation/types.py` (61c6257)
    - [ ] Write Tests
    - [ ] Implement Feature
- [x] Task: Refactor `src/evaluation/streaming_detector.py` to use the new interfaces (41b4cda)
    - [ ] Write Tests
    - [ ] Implement Feature
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Foundation and Interfaces' (Protocol in workflow.md)

## Phase 2: Stage Implementation
*Goal: Refactor existing Sentry and Judge logic into the new modular components.*

- [ ] Task: Implement `SentryInferenceStage` (MobileNetV3) as a modular component
    - [ ] Write Tests
    - [ ] Implement Feature
- [ ] Task: Implement `JudgeInferenceStage` (Wav2Vec 2.0) as a modular component
    - [ ] Write Tests
    - [ ] Implement Feature
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Stage Implementation' (Protocol in workflow.md)

## Phase 3: Benchmarking and Validation
*Goal: Implement performance measurement tools and verify the refactor.*

- [ ] Task: Create `src/evaluation/benchmarking.py` for latency and memory profiling
    - [ ] Write Tests
    - [ ] Implement Feature
- [ ] Task: Verify end-to-end cascade flow with the new modular architecture
    - [ ] Write Tests
    - [ ] Implement Feature
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Benchmarking and Validation' (Protocol in workflow.md)
