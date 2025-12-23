# Technology Stack - Wakeword Training Platform

## Core Languages & Runtimes
*   **Python 3.10+**: Primary programming language for training, UI, and data pipelines.
*   **CUDA 11.8+**: GPU acceleration for PyTorch training and inference.

## Deep Learning & AI
*   **PyTorch 2.1.2**: Main deep learning framework for model training.
*   **HuggingFace Transformers**: Source for pre-trained teacher models (Wav2Vec2) used in distillation.
*   **Conformer / Transformer**: Advanced hybrid architectures for robust teacher models.
*   **ONNX / ONNX Runtime**: Standardized model format for cross-platform inference and export.
*   **onnx2tf / TFLite**: Tools for converting models to edge-optimized TensorFlow Lite formats.
*   **Optuna**: Automated hyperparameter optimization (HPO).

## Audio Processing
*   **Librosa / Soundfile**: Core libraries for loading, analyzing, and transforming audio data.
*   **Torchaudio**: GPU-accelerated audio loading and augmentation.
*   **Sounddevice**: Real-time microphone input for streaming detection testing.
*   **Scipy**: Scientific computing used for signal processing and augmentations.

## Data & Configuration
*   **Numpy / Pandas**: Numerical computing and dataset management.
*   **Scikit-learn**: Machine learning utilities for metrics (EER, pAUC) and data splitting.
*   **Pydantic / PyYAML**: Strong typing, data validation, and hierarchical configuration management.

## UI & Visualization
*   **Gradio**: Framework for building the interactive web-based training dashboard.
*   **Plotly / Matplotlib / Seaborn**: Interactive and static data visualization for training metrics and audio features.
*   **Weights & Biases (wandb) / Weave / Tensorboard**: Experiment tracking, real-time visualization, and advanced call tracing with Weave.

## Infrastructure & Logging
*   **Structlog**: Structured, machine-readable logging for better observability.
*   **Memory Format Optimization**: `torch.channels_last` for enhanced Tensor Core utilization.
*   **Stability Metrics**: Implementation of EER (Equal Error Rate) and FAH (False Alarms per Hour) as core production-readiness indicators.
*   **Psutil**: System resource monitoring (CPU/GPU/RAM), used in benchmarking to profile memory usage.
*   **Psutil**: System resource monitoring (CPU/GPU/RAM), used in benchmarking to profile memory usage.
