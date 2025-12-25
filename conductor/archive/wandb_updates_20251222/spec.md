# Track Specification: Weights & Biases Maintenance

## Overview
Update the Weights & Biases integration to resolve deprecation warnings and explore 'Weave' integration for better tracing.

## Goals
- Resolve the `reinit=True` deprecation warning.
- Ensure clean run termination.
- (Optional) Investigate and optionally integrate `weave` if beneficial for the project scope.

## Functional Requirements
- Update `wandb.init` call in `src/training/wandb_callback.py`.
- Verify W&B logging still works after changes.

## Out of Scope
- Major changes to logging metrics.
