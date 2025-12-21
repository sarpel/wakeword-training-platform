# Plan: Refactor and Standardize Distributed Cascade Inference

## Phase 1: Foundation and Interfaces [checkpoint: fd1df77]
*Goal: Define the abstract interfaces and project structure for modular inference.*

- [x] Task: Define `InferenceEngine` and `StageBase` abstract classes in `src/evaluation/types.py` (61c6257)
    - [ ] Write Tests
    - [ ] Implement Feature
- [x] Task: Refactor `src/evaluation/streaming_detector.py` to use the new interfaces (41b4cda)
    - [ ] Write Tests
    - [ ] Implement Feature
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Foundation and Interfaces' (Protocol in workflow.md)

## Phase 2: Stage Implementation [checkpoint: 9ef8b1a]
*Goal: Refactor existing Sentry and Judge logic into the new modular components.*

- [x] Task: Implement `SentryInferenceStage` (MobileNetV3) as a modular component (dceff7c)
    - [ ] Write Tests
    - [ ] Implement Feature
- [x] Task: Implement `JudgeInferenceStage` (Wav2Vec 2.0) as a modular component (859dd02)
    - [ ] Write Tests
    - [ ] Implement Feature
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Stage Implementation' (Protocol in workflow.md)

## Phase 3: Benchmarking and Validation [checkpoint: b15b1a4]
*Goal: Implement performance measurement tools and verify the refactor.*

- [x] Task: Create `src/evaluation/benchmarking.py` for latency and memory profiling (14b5ebe)
    - [ ] Write Tests
    - [ ] Implement Feature
- [x] Task: Verify end-to-end cascade flow with the new modular architecture (07c6185)
    - [ ] Write Tests
    - [ ] Implement Feature
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Benchmarking and Validation' (Protocol in workflow.md)
