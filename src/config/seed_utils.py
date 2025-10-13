"""
Seed Setting and Determinism Utilities
Ensures reproducibility across runs
"""
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed value
        deterministic: If True, enables deterministic mode (slower but reproducible)
                      If False, enables benchmark mode (faster but non-deterministic)
    """
    # Python random
    random.seed(seed)

    # Numpy random
    np.random.seed(seed)

    # PyTorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Deterministic mode: reproducible but slower
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        logger.info(f"Seed set to {seed} with deterministic mode enabled")
    else:
        # Benchmark mode: faster but non-deterministic
        cudnn.benchmark = True
        cudnn.deterministic = False
        logger.info(f"Seed set to {seed} with benchmark mode enabled (non-deterministic)")

    logger.info(f"Random seed: {seed}")


def get_rng_state():
    """
    Get current random number generator state

    Returns:
        Dictionary containing RNG states
    """
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }


def set_rng_state(state: dict):
    """
    Restore random number generator state

    Args:
        state: Dictionary containing RNG states from get_rng_state()
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])

    if torch.cuda.is_available() and state['torch_cuda'] is not None:
        torch.cuda.set_rng_state_all(state['torch_cuda'])

    logger.info("RNG state restored")


if __name__ == "__main__":
    # Test seed setting
    print("Seed Utils Test")
    print("=" * 60)

    # Test deterministic mode
    print("\n1. Testing deterministic mode...")
    set_seed(42, deterministic=True)

    # Generate some random numbers
    rand_nums_1 = [random.random() for _ in range(5)]
    print(f"  Random numbers (run 1): {rand_nums_1}")

    # Reset seed and generate again
    set_seed(42, deterministic=True)
    rand_nums_2 = [random.random() for _ in range(5)]
    print(f"  Random numbers (run 2): {rand_nums_2}")

    if rand_nums_1 == rand_nums_2:
        print("  ✅ Deterministic mode works: results are identical")
    else:
        print("  ❌ Deterministic mode failed: results differ")

    # Test benchmark mode
    print("\n2. Testing benchmark mode...")
    set_seed(42, deterministic=False)
    print("  ✅ Benchmark mode enabled")

    # Test RNG state save/restore
    print("\n3. Testing RNG state save/restore...")
    set_seed(42)
    state = get_rng_state()
    rand_before = [random.random() for _ in range(3)]
    print(f"  Random numbers before restore: {rand_before}")

    # Generate more numbers
    [random.random() for _ in range(10)]

    # Restore state
    set_rng_state(state)
    rand_after = [random.random() for _ in range(3)]
    print(f"  Random numbers after restore: {rand_after}")

    if rand_before == rand_after:
        print("  ✅ RNG state save/restore works")
    else:
        print("  ❌ RNG state save/restore failed")

    print("\n✅ Seed utils module loaded successfully")
