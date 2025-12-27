import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd()))

from src.evaluation.evaluator import load_model_for_evaluation

# Mock paths if needed, or rely on src.config.paths
from src.config.paths import paths

try:
    print("Loading model...")
    checkpoint_path = paths.CHECKPOINTS / "best_model.pt"
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        sys.exit(0)  # Skip test if no model

    model, config = load_model_for_evaluation(checkpoint_path, device="cuda")
    model.cuda()
    model.eval()

    # 2. Check QAT
    has_qat = any("FakeQuantize" in m.__class__.__name__ for m in model.modules())
    print(f"Has QAT: {has_qat}")

    # 3. Simulate Inference with Autocast (like dataset_evaluator)
    print("Creating dummy input...")

    # Create dummy input. Shape depends on model.
    # TinyConv usually takes (B, 1, F, T)
    # Let's create a generic tensor that fits typical mel specs
    # (Batch, Channel, Freq, Time) -> (1, 1, 64, 50)
    inputs = torch.randn(1, 1, 64, 50).cuda()

    # My fix logic from dataset_evaluator.py:
    use_autocast = not has_qat
    if has_qat:
        inputs = inputs.float()

    print(f"Using autocast: {use_autocast}")
    print(f"Input type: {inputs.dtype}")

    with torch.cuda.amp.autocast(enabled=use_autocast):
        out = model(inputs)

    print("✅ Inference successful")
    print(f"Output shape: {out.shape}")

except Exception as e:
    import traceback

    traceback.print_exc()
    print(f"❌ Inference failed: {e}")
    sys.exit(1)
