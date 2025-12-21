# Spec: Visual Analysis & Debugging Suite

## Problem Statement
Users currently lack visibility into the internal decision-making of the Distributed Cascade. While we have modular Sentry and Judge components, there is no easy way for a user to visualize *why* a model failed or to tune thresholds based on real-world data without manually editing config files and rerunning scripts.

## Goals
- **Interactive Tuning:** Provide a Gradio-based interface to adjust Sentry/Judge thresholds and see immediate impact on metrics.
- **Deep-Dive Inspection:** Allow users to listen to and analyze specific audio samples that caused False Positives or False Negatives.
- **Production Observability:** Integrate benchmarking metrics (latency, memory) directly into the UI.

## Key Components
1. **Threshold Analysis Logic:** Backend to compute Precision-Recall curves and False Alarm rates on-the-fly.
2. **Analysis Dashboard (UI):** A new sub-panel in the Evaluation tab.
3. **Benchmarking Integration:** UI updates to `panel_training.py` and `panel_evaluation.py` to show execution metrics.

## Acceptance Criteria
- [ ] Users can adjust thresholds in the UI and see updated metrics.
- [ ] A "False Positive Gallery" allows inspecting and playing back error samples.
- [ ] Latency and memory metrics from `BenchmarkRunner` are displayed in the UI.
- [ ] All code follows project standards and has >80% test coverage.
