# Implementation Plan: Environment-Aware Orchestration & Notebooks

This plan outlines the steps to implement environment-specific optimizations, Docker Compose orchestration, and interactive notebooks, ensuring the platform runs optimally across Windows, WSL, and Cloud.

## Phase 1: Environment-Aware Configuration (.env Integration) [checkpoint: 9fb91d7]
- [x] Task: Implement .env-based feature toggles f4bf4b1
    - [x] Sub-task: Update `.env.example` with flags for `QUANTIZATION_BACKEND` (fbgemm/qnnpack), `MP_START_METHOD` (spawn/fork), and `USE_TRITON` (true/false).
    - [x] Sub-task: Write unit tests in `tests/test_env_config.py` to verify that configuration correctly resolves .env overrides.
    - [x] Sub-task: Implement config resolution logic in `src/config/` (or dedicated env util) to set PyTorch/OS defaults based on these flags.
    - [x] Sub-task: Verify that backend switching successfully updates the global PyTorch quantization config.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Environment-Aware Configuration' (Protocol in workflow.md)

## Phase 2: Docker Orchestration
- [ ] Task: Create Multi-Service Docker Orchestration
    - [ ] Sub-task: Refactor/Update root `Dockerfile` to support both the Gradio Dashboard and Jupyter Lab entry points.
    - [ ] Sub-task: Create `docker-compose.yml` defining `dashboard`, `inference-server`, and `jupyter-lab` services.
    - [ ] Sub-task: Configure NVIDIA GPU passthrough and persistent volume mappings for `/data`, `/models`, and `/logs`.
    - [ ] Sub-task: Write a test script `tests/test_docker_persistence.py` to verify that files written inside containers persist on the host.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Docker Orchestration' (Protocol in workflow.md)

## Phase 3: Interactive Notebooks
- [ ] Task: Create Interactive Entry Points
    - [ ] Sub-task: Develop `Jupyter_Quickstart.ipynb` at root for local dataset visualization and training triggers.
    - [ ] Sub-task: Develop `Colab_Training_Platform.ipynb` with blocks for `!pip install`, Google Drive mounting, and VRAM checks.
    - [ ] Sub-task: Verify notebook-to-script execution (ensuring the notebooks can correctly import `src` modules).
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Notebook Development' (Protocol in workflow.md)

## Phase 4: Final Integration & Documentation
- [ ] Task: Final Polish and Sync
    - [ ] Sub-task: Update `README.md` and `GUIDE.md` with instructions for Docker and Notebook usage.
    - [ ] Sub-task: Run full verification suite across all environments.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Final Integration' (Protocol in workflow.md)
