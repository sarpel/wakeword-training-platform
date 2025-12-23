# Initial Concept
A complete, user-friendly platform for training custom wakeword detection models with a web interface, featuring a distributed cascade architecture (Sentry, Judge, Teacher). Targeted at embedded developers and hobbyists for production-ready edge deployment.

# Product Guide - Wakeword Training Platform

## Vision
The Wakeword Training Platform is designed to democratize high-quality, production-ready wakeword detection. It bridges the gap between complex deep learning research and practical application by providing a user-friendly, "Google-tier" distributed cascade architecture that runs efficiently on everything from low-power edge devices to robust local servers.

## Target Users
*   **Embedded Systems Developers:** Professionals building voice-enabled products who require ultra-low-power, quantized models (MobileNetV3) that fit on microcontrollers and DSPs.
*   **Smart Home Hobbyists:** Enthusiasts seeking to add reliable, personalized custom voice triggers to their DIY projects without needing a PhD in machine learning.

## Core Value Proposition
*   **Professional Accuracy for Everyone:** Access to advanced techniques like Knowledge Distillation (The Teacher) and False Positive Rejection (The Judge) through a simple UI.
*   **End-to-End Pipeline:** From raw audio dataset management to one-click hardware-optimized exports.
*   **Transparency & Control:** Real-time feedback during training ensures users understand exactly how their model is performing before deployment.

## Key Features
*   **One-Click Export:** Seamlessly convert trained models to TFLite and ONNX formats, ready for integration into edge hardware.
*   **Visual Analysis & Debugging:** Comprehensive tools for visualizing model performance, analyzing false positives, and tuning detection thresholds.
*   **Closed-Loop Hard Negative Mining:** Interactive system to identify misclassified benchmarking samples and inject them back into training splits for iterative refinement.
*   **Standardized Benchmarking:** Built-in profiling tools to measure latency and memory usage across all stages of the distributed cascade.
*   **Robustness Engine:** Advanced acoustic simulation including Room Impulse Response (RIR), background noise with SNR scheduling, and pitch/speed perturbation.
*   **Precision Training:** Dual-teacher knowledge distillation (Wav2Vec2 + Conformer), intermediate feature alignment, and multi-objective optimization (pAUC vs Latency).
*   **Optimized Hyperparameter Tuning:** Multi-objective HPO using Optuna (NSGA-II) with exploit-and-explore mutation to discover ideal accuracy/speed trade-offs.
*   **Stable Streaming:** Integrated temporal smoothing and hysteresis logic to eliminate flickering detections in real-time usage.

## User Experience Goals
*   **Automated Simplicity:** Default presets that allow users to launch high-quality training pipelines with minimal manual configuration.
*   **Expert Granularity:** Deep-dive panels for advanced users to fine-tune hyper-parameters, loss functions, and distillation schedules.
*   **High-Performance Execution:** Hardware-aware optimizations (channels_last, non_blocking transfers) and real-time VRAM telemetry to maximize throughput on consumer-grade GPUs.
*   **Interactive HPO Results:** A structured, editable results table for HPO trials, allowing users to review, modify, and apply discovered configurations directly to their training session or save them as new profiles.
*   **Live Verification:** Real-time streaming detection and metric visualization to provide immediate confidence in model improvements.
