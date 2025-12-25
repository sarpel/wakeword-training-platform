# Specification: Environment-Aware Orchestration & Notebooks (multi_env_support_20251224)

## Overview
This track introduces environment-aware configurations to optimize performance across Windows, WSL/Linux, and Cloud (Colab) environments. It also provides a robust orchestration layer via Docker Compose and interactive entry points through Jupyter and Google Colab notebooks.

## Functional Requirements

### 1. Environment-Specific Logic & Feature Flags
- Implement a system to toggle features based on `.env` flags:
    - **Quantization Backend:** Choice between `fbgemm` (Windows/x86) and `qnnpack/xnnpack` (Linux/ARM).
    - **Optimizations:** Enable/disable `torch.compile` (Triton) and `NCCL` based on environment support.
    - **Multiprocessing:** Toggle between `spawn` (Windows) and `fork` (Linux) via configuration.
- Update `src/config/` to respect these `.env` variables during model initialization and data loading.

### 2. Docker Compose Orchestration
- Create a `docker-compose.yml` that orchestrates:
    - **Dashboard:** The Gradio-based training UI.
    - **Inference Server:** The FastAPI-based inference engine.
    - **Jupyter Lab:** For interactive development within the container.
- Implement persistent volume mapping for `/data`, `/models`, and `/logs`.
- Configure GPU pass-through for NVIDIA containers.

### 3. Interactive Notebooks
- **Jupyter Notebook:** A local guide for exploring datasets, visualizing audio, and running training experiments.
- **Google Colab Notebook:** A cloud-optimized version with automated dependency installation and Google Drive integration for model persistence.

## Non-Functional Requirements
- **Portability:** Ensure the project can be set up on a fresh machine (Windows or Linux) with minimal manual steps.
- **Resource Awareness:** Docker containers should have resource limits defined (CPU/RAM).

## Acceptance Criteria
- [ ] User can switch between `fbgemm` and `qnnpack` backends via a single change in `.env`.
- [ ] `docker-compose up` successfully starts the Dashboard, Inference Server, and Jupyter Lab.
- [ ] Changes made in the Docker container's `/data` directory persist on the host machine.
- [ ] Google Colab notebook successfully installs dependencies and runs a dummy training epoch.

## Out of Scope
- Support for non-NVIDIA GPUs (AMD/Intel) via DirectML in this specific track.
- Fully automated dataset downloading within the notebooks (user must still provide paths).
