# Track Specification: Model Excellence & Optimization Suite

## Overview
This track implements a suite of advanced training and optimization techniques to achieve the "theoretical best" performance for wakeword detection on a single-GPU setup. It focuses on maximizing precision through multi-objective HPO, refined knowledge distillation from dual teachers (Wav2Vec2 + one other), and an interactive hard negative mining loop.

## Goals
- Maximize the **pAUC (Partial Area Under Curve)** metric, targeting the high-precision region.
- Discover the "Pareto frontier" between accuracy and latency using sequential HPO.
- Implement a dual-teacher distillation protocol to robustly guide the student model.
- Reduce False Positive Rates through benchmarking-driven data mining.

## Functional Requirements

### 1. Advanced HPO (Hyperparameter Optimization)
- **Sequential Multi-Objective Engine:** Use Optuna (NSGA-II) to optimize for `pAUC` and `Latency` simultaneously.
- **Single-GPU Constraint:** Trials will run sequentially (one at a time) to ensure stability on single-GPU hardware.
- **PBT Elements:** Implementation of an "Exploit-and-Explore" scheduler to fork and mutate hyperparameters of top-performing trials within the sequential flow.

### 2. Knowledge Distillation 2.0 (Dual-Teacher)
- **Second Teacher Architecture:** Implement **Conformer** (or a similar high-performance CNN-Transformer hybrid) as the second teacher alongside Wav2Vec2.
- **Intermediate Feature Matching:** Loss functions to align student (MobileNetV3) feature maps with both teachers' representations.
- **Dynamic Temperature:** Automated temperature scheduling during the distillation process.

### 3. Interactive Hard Negative Mining
- **Benchmarking Integration:** A "Mine Hard Negatives" button in the Benchmark UI to identify misclassified samples.
- **Verification Queue:** A system to collect these samples for human "Confirm/Discard" verification.
- **Dataset Integration:** Automated tools to inject verified negatives back into the training split.

### 4. Training Pipeline Enhancements
- **Refined SpecAugment:** Time/frequency masking tailored for short wakeword utterances.
- **Noise Mixing Logic:** Advanced strategy for mixing background noise at varying SNR levels.

## Non-Functional Requirements
- **Single-GPU Optimization:** Ensure the training loop and distillation from two teachers fit within standard GPU VRAM (e.g., 8GB-12GB).
- **Reproducibility:** All optimization runs must be fully logged for resumption.

## Acceptance Criteria
- [ ] Sequential multi-objective HPO trial successfully generates a Pareto plot.
- [ ] Measured improvement in pAUC compared to baseline.
- [ ] Distillation pipeline successfully consumes signals from both Wav2Vec2 and the second teacher.
- [ ] Functional "Mining" button in the UI correctly queues samples for re-training.

## Out of Scope
- Parallel HPO trial execution (Multi-GPU/Distributed).
- Dynamic architecture search (NAS).
- More than two teacher models.
