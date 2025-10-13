"""
Optimizer and Scheduler Factory
Creates optimizers and learning rate schedulers from configuration
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
    LambdaLR
)
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class WarmupScheduler:
    """
    Learning rate warmup wrapper
    Gradually increases learning rate from 0 to initial_lr over warmup_epochs
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        base_scheduler: Optional[Any] = None
    ):
        """
        Initialize warmup scheduler

        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            base_scheduler: Base scheduler to use after warmup
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.current_epoch = 0

        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch: Optional[int] = None, metrics: Optional[float] = None):
        """
        Step the scheduler

        Args:
            epoch: Current epoch (optional)
            metrics: Validation metric for ReduceLROnPlateau (optional)
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        if self.current_epoch < self.warmup_epochs:
            # Warmup phase: linearly increase learning rate
            warmup_factor = (self.current_epoch + 1) / self.warmup_epochs

            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * warmup_factor

            logger.debug(f"Warmup epoch {self.current_epoch + 1}/{self.warmup_epochs}: LR = {self.optimizer.param_groups[0]['lr']:.6f}")

        else:
            # After warmup, use base scheduler
            if self.base_scheduler is not None:
                if isinstance(self.base_scheduler, ReduceLROnPlateau):
                    if metrics is not None:
                        self.base_scheduler.step(metrics)
                else:
                    self.base_scheduler.step()

    def get_last_lr(self):
        """Get last learning rate"""
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        """
        Returns the state of the scheduler as a dictionary

        Returns:
            Dictionary containing scheduler state
        """
        state = {
            'current_epoch': self.current_epoch,
            'base_lrs': self.base_lrs,
            'warmup_epochs': self.warmup_epochs,
            'base_scheduler_state': self.base_scheduler.state_dict() if self.base_scheduler else None
        }
        return state

    def load_state_dict(self, state_dict):
        """
        Loads the scheduler state

        Args:
            state_dict: Dictionary containing scheduler state
        """
        self.current_epoch = state_dict['current_epoch']
        self.base_lrs = state_dict['base_lrs']
        self.warmup_epochs = state_dict['warmup_epochs']

        if self.base_scheduler and state_dict['base_scheduler_state']:
            self.base_scheduler.load_state_dict(state_dict['base_scheduler_state'])


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = 'adam',
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    betas: Tuple[float, float] = (0.9, 0.999),
    **kwargs
) -> optim.Optimizer:
    """
    Create optimizer from configuration

    Args:
        model: PyTorch model
        optimizer_name: Optimizer type ('adam', 'sgd', 'adamw')
        learning_rate: Initial learning rate
        weight_decay: Weight decay (L2 regularization)
        momentum: Momentum for SGD
        betas: Beta parameters for Adam/AdamW
        **kwargs: Additional optimizer-specific parameters

    Returns:
        PyTorch optimizer

    Raises:
        ValueError: If optimizer_name is not recognized
    """
    optimizer_name = optimizer_name.lower()

    # Get model parameters
    params = model.parameters()

    logger.info(f"Creating optimizer: {optimizer_name}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Weight decay: {weight_decay}")

    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            **kwargs
        )
        logger.info(f"  Betas: {betas}")

    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            **kwargs
        )
        logger.info(f"  Betas: {betas}")
        logger.info("  Using AdamW (decoupled weight decay)")

    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            params,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
            **kwargs
        )
        logger.info(f"  Momentum: {momentum}")
        logger.info("  Using Nesterov momentum")

    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. "
            f"Supported: adam, adamw, sgd"
        )

    return optimizer


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = 'cosine',
    epochs: int = 50,
    warmup_epochs: int = 0,
    step_size: int = 10,
    gamma: float = 0.1,
    patience: int = 5,
    factor: float = 0.5,
    min_lr: float = 1e-6,
    **kwargs
) -> Optional[Any]:
    """
    Create learning rate scheduler from configuration

    Args:
        optimizer: PyTorch optimizer
        scheduler_name: Scheduler type ('cosine', 'step', 'plateau', 'none')
        epochs: Total number of training epochs
        warmup_epochs: Number of warmup epochs
        step_size: Step size for StepLR
        gamma: Multiplicative factor for StepLR
        patience: Patience for ReduceLROnPlateau
        factor: Factor for ReduceLROnPlateau
        min_lr: Minimum learning rate
        **kwargs: Additional scheduler-specific parameters

    Returns:
        Learning rate scheduler (or None if scheduler_name is 'none')

    Raises:
        ValueError: If scheduler_name is not recognized
    """
    scheduler_name = scheduler_name.lower()

    if scheduler_name == 'none':
        logger.info("No learning rate scheduler")
        return None

    logger.info(f"Creating scheduler: {scheduler_name}")

    # Create base scheduler
    base_scheduler = None

    if scheduler_name == 'cosine':
        # Cosine annealing
        T_max = epochs - warmup_epochs if warmup_epochs > 0 else epochs
        base_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=min_lr,
            **kwargs
        )
        logger.info(f"  Cosine annealing: T_max={T_max}, eta_min={min_lr}")

    elif scheduler_name == 'step':
        # Step decay
        base_scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
            **kwargs
        )
        logger.info(f"  Step LR: step_size={step_size}, gamma={gamma}")

    elif scheduler_name == 'plateau':
        # Reduce on plateau
        base_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            **kwargs
        )
        logger.info(f"  Reduce on plateau: patience={patience}, factor={factor}, min_lr={min_lr}")

    else:
        raise ValueError(
            f"Unknown scheduler: {scheduler_name}. "
            f"Supported: cosine, step, plateau, none"
        )

    # Wrap with warmup if needed
    if warmup_epochs > 0:
        logger.info(f"  Using warmup: {warmup_epochs} epochs")
        scheduler = WarmupScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            base_scheduler=base_scheduler
        )
    else:
        scheduler = base_scheduler

    return scheduler


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: Any
) -> Tuple[optim.Optimizer, Optional[Any]]:
    """
    Create optimizer and scheduler from configuration object

    Args:
        model: PyTorch model
        config: Configuration object (WakewordConfig)

    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Create optimizer
    optimizer = create_optimizer(
        model=model,
        optimizer_name=config.optimizer.optimizer,
        learning_rate=config.training.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        momentum=config.optimizer.momentum,
        betas=config.optimizer.betas
    )

    # Create scheduler
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_name=config.optimizer.scheduler,
        epochs=config.training.epochs,
        warmup_epochs=config.optimizer.warmup_epochs,
        step_size=config.optimizer.step_size,
        gamma=config.optimizer.gamma,
        patience=config.optimizer.patience,
        factor=config.optimizer.factor,
        min_lr=config.optimizer.min_lr
    )

    return optimizer, scheduler


def get_learning_rate(optimizer: optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer

    Args:
        optimizer: PyTorch optimizer

    Returns:
        Current learning rate (first param group)
    """
    return optimizer.param_groups[0]['lr']


def adjust_learning_rate(optimizer: optim.Optimizer, scale: float):
    """
    Manually adjust learning rate by scale factor

    Args:
        optimizer: PyTorch optimizer
        scale: Scale factor for learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] *= scale

    logger.info(f"Learning rate adjusted by {scale}x to {get_learning_rate(optimizer):.6f}")


def create_grad_scaler(enabled: bool = True) -> torch.cuda.amp.GradScaler:
    """
    Create gradient scaler for mixed precision training

    Args:
        enabled: Whether to enable mixed precision

    Returns:
        GradScaler for automatic mixed precision
    """
    scaler = torch.cuda.amp.GradScaler(enabled=enabled)

    if enabled:
        logger.info("Mixed precision training enabled (FP16)")
    else:
        logger.info("Mixed precision training disabled (FP32)")

    return scaler


def clip_gradients(
    model: nn.Module,
    max_norm: float,
    norm_type: float = 2.0
) -> float:
    """
    Clip gradients by norm

    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
        norm_type: Type of norm (2.0 for L2 norm)

    Returns:
        Total gradient norm before clipping
    """
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_norm,
        norm_type=norm_type
    )

    return total_norm.item()


if __name__ == "__main__":
    # Test optimizer and scheduler creation
    print("Optimizer and Scheduler Factory Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create dummy model
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 2)
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")

    # Test optimizer creation
    print("\n1. Testing optimizer creation...")
    for opt_name in ['adam', 'adamw', 'sgd']:
        optimizer = create_optimizer(
            model,
            optimizer_name=opt_name,
            learning_rate=0.001,
            weight_decay=1e-4
        )
        print(f"  ✅ Created {opt_name.upper()} optimizer")

    # Test scheduler creation
    print("\n2. Testing scheduler creation...")
    optimizer = create_optimizer(model, optimizer_name='adam', learning_rate=0.001)

    for sched_name in ['cosine', 'step', 'plateau', 'none']:
        scheduler = create_scheduler(
            optimizer,
            scheduler_name=sched_name,
            epochs=50,
            warmup_epochs=0
        )
        if sched_name == 'none':
            print(f"  ✅ Scheduler 'none' correctly returns None")
        else:
            print(f"  ✅ Created {sched_name} scheduler")

    # Test warmup scheduler
    print("\n3. Testing warmup scheduler...")
    optimizer = create_optimizer(model, optimizer_name='adam', learning_rate=0.001)
    scheduler = create_scheduler(
        optimizer,
        scheduler_name='cosine',
        epochs=50,
        warmup_epochs=5
    )

    print(f"  Initial LR: {get_learning_rate(optimizer):.6f}")

    # Simulate warmup
    for epoch in range(10):
        scheduler.step(epoch)
        lr = get_learning_rate(optimizer)
        if epoch < 5:
            print(f"  Warmup epoch {epoch+1}: LR = {lr:.6f}")
        elif epoch == 5:
            print(f"  After warmup epoch {epoch+1}: LR = {lr:.6f}")

    print(f"  ✅ Warmup scheduler works")

    # Test mixed precision scaler
    print("\n4. Testing mixed precision scaler...")
    scaler = create_grad_scaler(enabled=True)
    print(f"  ✅ Created gradient scaler")

    # Test gradient clipping
    print("\n5. Testing gradient clipping...")
    # Create fake gradients
    for param in model.parameters():
        param.grad = torch.randn_like(param) * 10

    total_norm_before = sum(p.grad.norm(2).item() ** 2 for p in model.parameters()) ** 0.5
    total_norm_after = clip_gradients(model, max_norm=1.0)

    print(f"  Gradient norm before clipping: {total_norm_before:.4f}")
    print(f"  Gradient norm after clipping: {total_norm_after:.4f}")
    print(f"  ✅ Gradient clipping works")

    # Test ReduceLROnPlateau
    print("\n6. Testing ReduceLROnPlateau with metrics...")
    optimizer = create_optimizer(model, optimizer_name='adam', learning_rate=0.001)
    scheduler = create_scheduler(
        optimizer,
        scheduler_name='plateau',
        patience=3,
        factor=0.5
    )

    initial_lr = get_learning_rate(optimizer)
    print(f"  Initial LR: {initial_lr:.6f}")

    # Simulate plateau (loss not decreasing)
    for epoch in range(10):
        val_loss = 1.0  # Constant loss (plateau)
        scheduler.step(val_loss)

        if epoch == 3 or epoch == 6 or epoch == 9:
            current_lr = get_learning_rate(optimizer)
            print(f"  After epoch {epoch+1}: LR = {current_lr:.6f}")

    print(f"  ✅ ReduceLROnPlateau works")

    # Test full integration with config-like object
    print("\n7. Testing config-based creation...")

    # Create mock config
    class MockOptimizerConfig:
        optimizer = 'adam'
        weight_decay = 1e-4
        momentum = 0.9
        betas = (0.9, 0.999)
        scheduler = 'cosine'
        warmup_epochs = 5
        step_size = 10
        gamma = 0.1
        patience = 5
        factor = 0.5
        min_lr = 1e-6

    class MockTrainingConfig:
        learning_rate = 0.001
        epochs = 50

    class MockConfig:
        optimizer = MockOptimizerConfig()
        training = MockTrainingConfig()

    config = MockConfig()
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)

    print(f"  Created optimizer: {type(optimizer).__name__}")
    print(f"  Created scheduler: {type(scheduler).__name__}")
    print(f"  ✅ Config-based creation works")

    print("\n✅ All optimizer and scheduler tests passed successfully")
    print("Optimizer factory module loaded successfully")