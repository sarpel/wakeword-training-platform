# AGENTS.md - AI Coding Agent Guidelines

> **Project**: Wakeword Training Platform  
> **Python**: 3.10+  
> **Framework**: PyTorch + Gradio

---

## Quick Reference: Commands

### Running the Application
```bash
python run.py                    # Launch Gradio UI at http://localhost:7860
make run                         # Same as above
uvicorn server.app:app --reload  # Inference server at :8000
```

### Testing
```bash
# Run all tests
pytest tests/ -v --tb=short
make test

# Run a SINGLE test file
pytest tests/test_config.py -v

# Run a SINGLE test function
pytest tests/test_config.py::TestDefaultConfig::test_default_config_creation -v

# Run tests by marker
pytest -m unit           # Fast unit tests
pytest -m integration    # Integration tests
pytest -m gpu            # GPU-required tests (auto-skipped if no CUDA)
pytest -m slow           # Long-running tests

# With coverage
pytest --cov=src --cov-report=term-missing --cov-fail-under=80
make test-cov
```

### Linting & Formatting
```bash
# Check (no changes)
make lint                # Runs flake8, isort --check, black --check

# Auto-fix
make format              # Runs isort + black

# Individual tools
black src/ tests/ --line-length=120
isort src/ tests/ --profile=black
flake8 src/ tests/ --max-line-length=120
mypy src/ --ignore-missing-imports
```

### Installation
```bash
pip install -e ".[dev]"      # Dev dependencies
make install-dev             # Same + pre-commit hooks
make install-gpu             # PyTorch CUDA 11.8
```

---

## Code Style Guidelines

### Formatting Rules
- **Line length**: 120 characters (Black + flake8)
- **Formatter**: Black (non-negotiable)
- **Import sorting**: isort with `profile = "black"`
- **Quote style**: Double quotes `"string"` (Black default)

### Import Order (isort sections)
```python
# 1. Future
from __future__ import annotations

# 2. Standard library
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 3. Third-party
import numpy as np
import torch
import torch.nn as nn

# 4. First-party (src.*)
from src.config.defaults import WakewordConfig
from src.config.logger import get_logger
```

### Type Hints
- **Required** on all function signatures
- Use `Optional[X]` for nullable types
- Use `TYPE_CHECKING` guard for circular imports:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.defaults import WakewordConfig
```

### Docstrings (Google Style)
```python
def train_model(config: WakewordConfig, epochs: int = 10) -> float:
    """
    Train the wakeword detection model.

    Args:
        config: Training configuration object.
        epochs: Number of training epochs.

    Returns:
        Final validation loss.

    Raises:
        ConfigurationError: If config is invalid.
    """
```

### Logging
```python
import structlog
logger = structlog.get_logger(__name__)

# Usage
logger.info("Training started", epoch=1, lr=0.001)
logger.error("Failed to load", path=str(file_path), exc_info=True)
```

### Path Handling
```python
from pathlib import Path

# Always use pathlib, never os.path
data_dir = Path("data")
model_path = data_dir / "models" / "best_model.pt"
model_path.parent.mkdir(parents=True, exist_ok=True)
```

### Configuration
- Use `src.config.defaults` dataclasses (NOT raw dicts/hardcoded values)
- Access via: `config.training.learning_rate`
```python
from src.config.defaults import WakewordConfig
config = WakewordConfig()
```

### Error Handling
- Use custom exceptions from `src.exceptions`:
```python
from src.exceptions import ConfigurationError, DataLoadError

if not path.exists():
    raise DataLoadError(f"File not found: {path}")
```

---

## Test Conventions

### File Naming
- Test files: `tests/test_*.py`
- Test classes: `class Test*:`
- Test functions: `def test_*():`

### Markers (use in every test)
```python
@pytest.mark.unit          # Fast, no I/O, no GPU
@pytest.mark.integration   # May need files/GPU
@pytest.mark.slow          # > 5 seconds
@pytest.mark.gpu           # Requires CUDA
```

### Fixtures (from conftest.py)
```python
def test_example(default_config, device, sample_audio, tmp_path):
    # default_config: WakewordConfig instance
    # device: "cuda" or "cpu"
    # sample_audio: np.ndarray (1.5s @ 16kHz)
    # tmp_path: pytest temporary directory
    pass
```

---

## Project Structure

```
src/
  config/       # Pydantic configs, logger, CUDA utils
  data/         # Dataset, augmentation, audio processing
  evaluation/   # Metrics, benchmarking, inference
  export/       # ONNX/TFLite exporters
  models/       # Architectures (ResNet, MobileNet, LSTM)
  training/     # Trainer, HPO, distillation, checkpoints
  ui/           # Gradio panels
tests/          # Mirrors src/ structure
```

---

## Critical Patterns

### GPU/Device Handling
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
tensor = tensor.to(device)
```

### Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = model(inputs)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Checkpoint Loading
```python
from src.training.checkpoint_manager import CheckpointManager
manager = CheckpointManager(checkpoint_dir)
state = manager.load_checkpoint("best_model.pt")
```

---

## Common Pitfalls

1. **Never hardcode hyperparameters** - Use `config.*` attributes
2. **Never use `os.path`** - Use `pathlib.Path`
3. **Always mark tests with markers** - `@pytest.mark.unit` etc.
4. **Respect 120 char line limit** - Black enforces this
5. **Type hint return values** - Even if just `-> None`
6. **Use structlog, not print()** - For all logging

---

## Pre-commit Hooks (Enabled)

On every commit, these run automatically:
- `trailing-whitespace` - Removes trailing spaces
- `end-of-file-fixer` - Ensures newline at EOF
- `black` - Code formatting
- `isort` - Import sorting
- `detect-secrets` - Blocks committed secrets
- `bandit` - Security scanning

If commit fails, run `make format` then retry.

---

## Environment Variables

Key env vars (see `.env.example`):
- `QUANTIZATION_BACKEND`: `fbgemm` (Windows) or `qnnpack` (Linux)
- `TRAINING_NUM_WORKERS`: DataLoader workers (default: 8)
- `CUDA_VISIBLE_DEVICES`: GPU selection

---

## Copilot/Agent Instructions (from .github/copilot-instructions.md)

1. **Entry Point**: `run.py` launches `src/ui/app.py`
2. **Data Pipeline**: `src/data/` handles audio loading, CMVN, augmentation
3. **Training**: Custom loop with AMP, gradient clipping, EMA
4. **Config**: Pydantic models in `src/config/pydantic_validator.py`
5. **Metrics**: FAH (False Alarms/Hour), EER, pAUC in `src/evaluation/`

When modifying core logic, verify with a reproduction script since the test suite is comprehensive but domain-specific.
