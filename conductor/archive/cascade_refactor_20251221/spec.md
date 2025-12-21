# Spec: Refactor and Standardize Distributed Cascade Inference

## Problem Statement
The current "Distributed Cascade Architecture" (Sentry, Judge, Teacher) lacks a standardized, modular inference interface. This makes it difficult to swap models, benchmark latency accurately across different stages, and ensure that the "Sentry" (Edge) logic remains lightweight and decoupled from the "Judge" (Local Server) logic.

## Goals
- **Strict Modularity:** Define abstract base classes for Sentry and Judge components.
- **Standardized Inference API:** Ensure both components follow a unified `predict(audio_segment)` interface.
- **Benchmarking Suite:** Implement a standard tool to measure latency and memory usage for each stage.
- **Clean Decoupling:** Ensure Sentry logic can run independently without Judge dependencies.

## Key Components
1. **`SentryBase` / `JudgeBase`:** Abstract classes defining the interface.
2. **`StreamingInferenceEngine`:** A unified engine that manages the handoff between Sentry and Judge.
3. **`BenchmarkRunner`:** A utility for measuring performance metrics.

## Acceptance Criteria
- [ ] Sentry and Judge logic refactored into modular components.
- [ ] All unit tests pass with >80% coverage.
- [ ] Latency benchmarking script successfully measures execution time for both stages.
- [ ] No functional regression in wakeword detection accuracy.
