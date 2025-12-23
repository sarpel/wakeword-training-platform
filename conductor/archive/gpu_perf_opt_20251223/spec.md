# Specification - Track: High-Performance GPU Training Pipeline (5.5GB VRAM)

## Overview
This track focuses on maximizing training throughput for a 5.5GB VRAM environment by migrating all computational components (Teacher & Student) to the GPU and implementing zero-impact performance optimizations (`channels_last`, `non_blocking`, `persistent_workers`).

## Functional Requirements

### 1. GPU Migration & Device Strategy
- **Unified GPU Residence:** Explicitly configure the Teacher model(s) to load on `cuda` by default.
- **Async Transfers:** Implement `non_blocking=True` in all `tensor.to(device)` calls within the training loop to overlap CPU-to-GPU data movement with computation.
- **Persistent Data Loading:** Enable `persistent_workers=True` in DataLoaders to eliminate the process spawn overhead at the start of every epoch.

### 2. Memory & Architecture Optimization
- **Channels Last Format:** Convert both the Student and Teacher models to `memory_format=torch.channels_last`. This optimizes memory access patterns for NVIDIA Tensor Cores, providing a throughput boost with zero impact on numerical accuracy.
- **VRAM Telemetry:** Integrate real-time VRAM monitoring into the training dashboard to help users stay within the 5.5GB ceiling.

### 3. Stability & Safety
- **Pre-flight Estimation:** Log the estimated VRAM footprint of the teacher model during initialization to warn users before a potential OOM.
- **Graceful OOM Recovery:** Catch `RuntimeError: CUDA out of memory` and provide actionable UI feedback (e.g., suggesting batch size reduction).

## Non-Functional Requirements
- **Accuracy Parity:** All optimizations must be numerically equivalent to existing implementations; model quality must not be affected.
- **Latency reduction:** Aim for a 2x-4x increase in samples/second throughput during Distillation.

## Acceptance Criteria
- [ ] Training log shows "Using channels_last memory format" and "Non-blocking transfers enabled".
- [ ] Teacher model successfully executes on GPU alongside the Student.
- [ ] UI displays active GPU memory usage during training.
- [ ] No regression in validation accuracy compared to current baseline.

## Out of Scope
- Implementation of smaller student architectures.
- Multi-GPU or Distributed Data Parallel (DDP) support.
