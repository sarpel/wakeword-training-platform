"""
Exponential Moving Average (EMA) for model parameters
Maintains shadow copy of model weights for more stable inference
"""

import time

from typing import Any, Dict, Optional, Tuple, cast



import structlog
import torch
import torch.nn as nn

logger = structlog.get_logger(__name__)


class EMA:
    """
    Exponential Moving Average of model parameters

    Maintains a shadow copy of model weights that are updated with:
        ema_weight = decay * ema_weight + (1 - decay) * model_weight

    Higher decay (closer to 1.0) = slower adaptation
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, device: Optional[str] = None):
        """
        Initialize EMA

        Args:
            model: Model to track
            decay: Decay rate (typically 0.999 or 0.9995)
            device: Device to store shadow parameters
        """
        self.model = model
        self.decay = decay
        self.device = device if device else next(model.parameters()).device

        # Create shadow copy of parameters
        self.shadow_params: Dict[str, torch.Tensor] = {}
        self._register_params()

        # Step counter
        self.num_updates = 0

        logger.info(f"EMA initialized with decay={decay}")

    def _register_params(self) -> None:
        """Register all model parameters for tracking"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Clone parameter and move to device
                # Mypy: Explicitly cast device to ensure proper .to() overload
                target_device = cast(Any, self.device)
                self.shadow_params[name] = param.data.clone().to(device=target_device)

    def update(self, decay: Optional[float] = None) -> None:
        """
        Update shadow parameters

        Args:
            decay: Optional override decay rate
        """
        current_decay = decay if decay is not None else self.decay

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    # EMA update: shadow = decay * shadow + (1-decay) * current
                    self.shadow_params[name].mul_(current_decay).add_(param.data, alpha=(1.0 - current_decay))

        self.num_updates += 1

    def apply_shadow(self) -> Dict[str, torch.Tensor]:
        """
        Apply shadow parameters to model (for inference/evaluation)

        Returns the original parameters for restoration
        """
        original_params = {}

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.shadow_params:
                    # Store original
                    original_params[name] = param.data.clone()

                    # Apply shadow
                    param.data.copy_(self.shadow_params[name])

        return original_params

    def restore(self, original_params: Dict[str, torch.Tensor]) -> None:
        """
        Restore original parameters after evaluation

        Args:
            original_params: Dictionary of original parameters
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in original_params:
                    param.data.copy_(original_params[name])

    def state_dict(self) -> Dict[str, Any]:
        """Get EMA state dict for checkpointing"""
        return {
            "shadow_params": self.shadow_params,
            "decay": self.decay,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load EMA state from checkpoint"""
        self.shadow_params = state_dict["shadow_params"]
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]


class EMAScheduler:
    """
    Adaptive EMA decay scheduler

    Adjusts decay rate based on training progress
    """

    def __init__(
        self,
        ema: EMA,
        initial_decay: float = 0.999,
        final_decay: float = 0.9995,
        warmup_epochs: int = 0,
        final_epochs: int = 10,
    ):
        """
        Initialize EMA scheduler

        Args:
            ema: EMA object to control
            initial_decay: Starting decay rate
            final_decay: Final decay rate (for last epochs)
            warmup_epochs: Number of warmup epochs with initial decay
            final_epochs: Number of final epochs to use final decay
        """
        self.ema = ema
        self.initial_decay = initial_decay
        self.final_decay = final_decay
        self.warmup_epochs = warmup_epochs
        self.final_epochs = final_epochs

        logger.info(
            f"EMA Scheduler: "
            f"initial={initial_decay}, final={final_decay}, "
            f"warmup={warmup_epochs}, final_epochs={final_epochs}"
        )

    def step(self, epoch: int, total_epochs: int) -> float:
        """
        Get decay rate for current epoch

        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of epochs

        Returns:
            Decay rate for this epoch
        """
        if epoch < self.warmup_epochs:
            # Warmup phase: use initial decay
            decay = self.initial_decay

        elif epoch >= (total_epochs - self.final_epochs):
            # Final phase: use final decay (higher = more stable)
            decay = self.final_decay

        else:
            # Middle phase: interpolate
            progress = (epoch - self.warmup_epochs) / max((total_epochs - self.final_epochs - self.warmup_epochs), 1)
            decay = self.initial_decay + (self.final_decay - self.initial_decay) * progress

        # Update EMA decay
        self.ema.decay = decay

        return decay


def create_ema(
    model: nn.Module,
    decay: float = 0.999,
    use_scheduler: bool = True,
    total_epochs: Optional[int] = None,
) -> Tuple[EMA, Optional[EMAScheduler]]:
    """
    Create EMA and optional scheduler

    Args:
        model: Model to track
        decay: Initial decay rate
        use_scheduler: Whether to use adaptive scheduler
        total_epochs: Total epochs (required if use_scheduler=True)

    Returns:
        Tuple of (ema, scheduler) or (ema, None)
    """
    ema = EMA(model, decay=decay)

    if use_scheduler:
        if total_epochs is None:
            raise ValueError("total_epochs required for scheduler")

        scheduler = EMAScheduler(
            ema,
            initial_decay=decay,
            final_decay=0.9995,
            warmup_epochs=0,
            final_epochs=10,
        )

        return ema, scheduler

    return ema, None


if __name__ == "__main__":
    # Test EMA
    print("EMA Test")
    print("=" * 60)

    # Create dummy model
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 2))

    print(f"Created test model with {sum(p.numel() for p in model.parameters())} parameters")

    # Get initial weights
    initial_weight = model[0].weight.data.clone()

    # Create EMA
    ema = EMA(model, decay=0.999)
    print("\nEMA created with decay=0.999")

    # Simulate training updates
    print("\nSimulating 100 training steps...")

    for step in range(100):
        # Simulate gradient update (modify model weights)
        with torch.no_grad():
            for param in model.parameters():
                param.data.add_(torch.randn_like(param) * 0.01)

        # Update EMA
        ema.update()

    final_weight = model[0].weight.data.clone()
    shadow_weight = ema.shadow_params["0.weight"]

    print("Weight changes:")
    print(f"  Initial weight mean: {initial_weight.mean():.6f}")
    print(f"  Final weight mean: {final_weight.mean():.6f}")
    print(f"  Shadow weight mean: {shadow_weight.mean():.6f}")

    # Test apply/restore
    print("\nTesting apply_shadow and restore...")

    original_params = ema.apply_shadow()
    applied_weight = model[0].weight.data.clone()

    assert torch.allclose(applied_weight, shadow_weight), "Shadow not applied correctly"
    print("  ✅ Shadow applied correctly")

    ema.restore(original_params)
    restored_weight = model[0].weight.data.clone()

    assert torch.allclose(restored_weight, final_weight), "Parameters not restored correctly"
    print("  ✅ Parameters restored correctly")

    # Test scheduler
    print("\nTesting EMA Scheduler...")

    ema2 = EMA(model, decay=0.999)
    scheduler = EMAScheduler(ema2, initial_decay=0.999, final_decay=0.9995, warmup_epochs=5, final_epochs=10)

    total_epochs = 50
    print(f"  Total epochs: {total_epochs}")
    print("  Decay schedule:")

    for epoch in [0, 4, 5, 20, 39, 40, 49]:
        decay = scheduler.step(epoch, total_epochs)
        print(f"    Epoch {epoch:2d}: decay={decay:.5f}")

    # Test state dict
    print("\nTesting state_dict save/load...")

    state = ema.state_dict()
    print(f"  State dict keys: {list(state.keys())}")
    print(f"  Num updates: {state['num_updates']}")

    ema3 = EMA(model, decay=0.5)  # Different decay
    ema3.load_state_dict(state)

    assert ema3.decay == ema.decay, "Decay not loaded correctly"
    assert ema3.num_updates == ema.num_updates, "Num updates not loaded correctly"
    print("  ✅ State dict save/load works")

    print("\n✅ All EMA tests passed")
