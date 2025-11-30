# Project: Wakeword-Platform

## Overview
A production-ready, GPU-accelerated platform for training custom wakeword detection models (e.g., "Hey Siri", "Alexa"). It features a comprehensive web interface (Gradio) handling the entire lifecycle: dataset management, configuration, training, evaluation, and export.

## Scope
End-to-end solution for creating lightweight, high-accuracy wakeword models suitable for edge deployment.

## Key Features
- **Web Interface**: 6-panel Gradio UI for accessible model development.
- **Training**: GPU-accelerated (CUDA), Mixed Precision (FP16), CMVN, EMA, Balanced Sampling.
- **Evaluation**: Real-time microphone testing, Streaming detection, Advanced metrics (FAH, EER, ROC).
- **Export**: ONNX, TorchScript, Quantized INT8.
- **Data Management**: Auto-splitting, Dataset scanning, Pre-computed feature caching.

## Tech Stack
- **Language**: Python 3.8+
- **Deep Learning**: PyTorch 2.1.2 + CUDA 11.8
- **UI**: Gradio 4.44
- **Audio**: Librosa, SoundFile, SoundDevice
- **Config/Validation**: Pydantic, PyYAML
- **Logging**: Structlog
- **Deployment**: ONNX Runtime

## Development Roadmap
### Phase 1: Foundation & Stability (Current Focus)
- [x] Pydantic Migration (Partially complete)
- [ ] **Refactor Dataset Logic**: Dynamic labels & explicit fallback (Immediate Task)
- [ ] Centralized Path Management
- [ ] Comprehensive Error Handling

### Phase 2: Code Quality & Performance
- [ ] Refactor Monolithic Trainer
- [ ] Optimize DataLoader (persistent workers, prefetch)
- [ ] Full Type Hinting (`mypy`)
- [ ] Standardize Structlog usage

### Phase 3: Advanced Features
- [ ] Experiment Tracking (WandB/MLflow)
- [ ] Hyperparameter Optimization (Optuna)
- [ ] Docker Support

## Key Technical Decisions
- **Strict CUDA Requirement**: Project enforces NVIDIA GPU availability for performance.
- **Architecture**: 
  - **UI**: Pure presentation layer (`src/ui`), decoupled from logic.
  - **Config**: `dataclasses` + `Pydantic` for strict validation.
  - **Data**: CPU-based loading/preprocessing -> GPU-based training.
- **Augmentation Strategy**: Heavy time-domain (CPU) + Lightweight frequency-domain (GPU/SpecAugment).

## Metrics
- **Primary**: F1 Score, Accuracy
- **Production**: False Alarms per Hour (FAH), False Positive Rate (FPR) at high Recall
- **Research**: Equal Error Rate (EER), ROC-AUC

## Risks & Constraints
- **Hardware**: Strictly requires NVIDIA GPU (no CPU/MPS training support yet).
- **Data**: Relies on user-provided dataset quality; imbalance handling is critical.
- **Complexity**: "God object" tendencies in `Trainer` and `Dataset` classes need refactoring.

## Next Steps (Immediate)
1.  Implement `TECHNICAL_DESIGN_DATA_REFACTOR.md`:
    -   Inject dynamic label mapping into `WakewordDataset`.
    -   Fix `fallback_to_audio` logic to be strict.
    -   Correct docstrings in `FeatureExtractor`.
