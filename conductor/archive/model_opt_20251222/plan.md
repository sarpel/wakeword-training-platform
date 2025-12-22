# Implementation Plan: Production Readiness & Model Optimization

This plan follows the Test-Driven Development (TDD) methodology and the specific phase completion protocols defined in `conductor/workflow.md`.

## Phase 1: Advanced Data Augmentation & Robustness [checkpoint: 4c20346]
Focus on improving the model's ability to handle real-world acoustic environments.

- [x] Task: Write tests for RIR, pitch, and speed perturbation in `tests/test_augmentation_advanced.py` [1685694]
- [x] Task: Implement RIR (Room Impulse Response) simulation in `src/data/augmentation.py` [47e1aaf]
- [x] Task: Implement Pitch and Speed perturbation in `src/data/augmentation.py` [7b76264]
- [x] Task: Update dataset loading to support these new augmentation types [2a40fd3]
- [x] Task: Conductor - User Manual Verification 'Phase 1: Advanced Augmentation' (Protocol in workflow.md)

## Phase 2: Specialized Loss Functions & Class Imbalance [checkpoint: a4503fb]
Optimize the training objective to handle the 90% negative class imbalance.

- [x] Task: Write tests for Focal Loss and Weighted Cross-Entropy in `tests/test_losses_advanced.py` [8f5a806]
- [x] Task: Implement Focal Loss in `src/training/losses.py` [8f5a806]
- [x] Task: Implement Weighted Cross-Entropy support in `src/training/trainer.py` [463678b]
- [x] Task: Integrate loss selection into the training configuration and UI [536d84d]
- [x] Task: Conductor - User Manual Verification 'Phase 2: Loss Optimization' (Protocol in workflow.md)

## Phase 3: Knowledge Distillation Pipeline [checkpoint: 8b947a4]
Implement the Teacher-Student training flow to boost edge model performance.

- [x] Task: Write tests for Teacher-Student logit matching in `tests/test_distillation_pipeline.py` [3bbf12b]
- [x] Task: Refine the `Teacher` model loading and inference logic in `src/models/distillation.py` [3bbf12b]
- [x] Task: Implement the distillation training loop (Kullback-Leibler divergence on soft labels) [3bbf12b]
- [x] Task: Verify the distilled student outperforms the baseline student [36bedb3]
- [x] Task: Conductor - User Manual Verification 'Phase 3: Knowledge Distillation' (Protocol in workflow.md)

## Phase 4: Hard Negative Mining & Streaming Refinement [checkpoint: f0f39e8]
Close the loop on false alarms and stabilize real-time detection.

- [x] Task: Create a utility script to extract high-confidence False Positives from evaluation logs [6485130]
- [x] Task: Implement temporal smoothing logic (N-of-M frames) in the streaming detector [0904386]
- [x] Task: Conduct a final end-to-end evaluation to verify FNR reduction and FAH maintenance [0904386]
- [x] Task: Conductor - User Manual Verification 'Phase 4: Final Production Tuning' (Protocol in workflow.md)
