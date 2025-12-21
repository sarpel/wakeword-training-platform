"""
Learning Rate Finder
Implements LR range test to find optimal learning rate
"""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import structlog
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = structlog.get_logger(__name__)


class LRFinder:
    """
    Learning Rate Finder using exponential LR increase

    Based on "Cyclical Learning Rates for Training Neural Networks" (Smith, 2017)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cuda",
    ):
        """
        Initialize LR Finder

        Args:
            model: Model to train
            optimizer: Optimizer (will be reset after finding)
            criterion: Loss function
            device: Device for training
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # Save initial state
        self.model_state = model.state_dict()
        self.optimizer_state = optimizer.state_dict()

        # Results
        self.learning_rates: List[float] = []
        self.losses: List[float] = []

    def range_test(
        self,
        train_loader: DataLoader,
        start_lr: float = 1e-6,
        end_lr: float = 1e-2,
        num_iter: int = 200,
        smooth_f: float = 0.05,
        diverge_th: float = 5.0,
    ) -> Tuple[List[float], List[float]]:
        """
        Run LR range test

        Args:
            train_loader: Training data loader
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_iter: Number of iterations
            smooth_f: Smoothing factor for loss
            diverge_th: Stop if loss exceeds best_loss * diverge_th

        Returns:
            Tuple of (learning_rates, losses)
        """
        logger.info(f"Running LR finder: " f"start_lr={start_lr:.2e}, end_lr={end_lr:.2e}, " f"num_iter={num_iter}")

        # Reset to initial state
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

        # Set to training mode
        self.model.train()

        # Calculate LR multiplier
        lr_mult = (end_lr / start_lr) ** (1 / num_iter)

        # Set initial LR
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = start_lr

        # Tracking
        best_loss = float("inf")
        smoothed_loss = 0

        # Iterate
        iterator = iter(train_loader)
        pbar = tqdm(range(num_iter), desc="LR Finder")

        for iteration in pbar:
            # Get batch
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)

            # Unpack batch
            if len(batch) == 3:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Check for divergence
            if iteration > 0 and loss.item() > diverge_th * best_loss:
                logger.info(f"Stopping early: loss diverged at iteration {iteration}")
                break

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Get current LR
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Smooth loss
            if iteration == 0:
                smoothed_loss = loss.item()
            else:
                smoothed_loss = smooth_f * loss.item() + (1 - smooth_f) * smoothed_loss

            # Track best loss
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

            # Record
            self.learning_rates.append(current_lr)
            self.losses.append(smoothed_loss)

            # Update progress bar
            pbar.set_postfix({"lr": f"{current_lr:.2e}", "loss": f"{smoothed_loss:.4f}"})

            # Increase LR
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= lr_mult

        logger.info(f"LR finder complete: {len(self.learning_rates)} iterations")

        # Restore initial state
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

        return self.learning_rates, self.losses

    def plot(self, skip_start: int = 10, skip_end: int = 5, save_path: Optional[Path] = None) -> None:
        """
        Plot LR vs Loss

        Args:
            skip_start: Skip first N points
            skip_end: Skip last N points
            save_path: Path to save plot
        """
        if len(self.learning_rates) == 0:
            logger.warning("No data to plot. Run range_test() first.")
            return

        # Prepare data
        lrs = np.array(self.learning_rates[skip_start : -skip_end if skip_end > 0 else None])
        losses = np.array(self.losses[skip_start : -skip_end if skip_end > 0 else None])

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(lrs, losses)
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Learning Rate Finder", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Find steepest decline
        # Calculate gradient
        grad = np.gradient(losses)
        min_grad_idx = grad.argmin()

        suggested_lr = lrs[min_grad_idx]
        ax.axvline(
            suggested_lr,
            color="r",
            linestyle="--",
            label=f"Suggested LR: {suggested_lr:.2e}",
        )
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Plot saved to {save_path}")

        plt.show()

    def suggest_lr(self, skip_start: int = 10, skip_end: int = 5) -> float:
        """
        Suggest optimal learning rate

        Uses the point of steepest descent

        Args:
            skip_start: Skip first N points
            skip_end: Skip last N points

        Returns:
            Suggested learning rate
        """
        if len(self.learning_rates) == 0:
            logger.warning("No data available. Run range_test() first.")
            return 0.001

        # Prepare data
        lrs = np.array(self.learning_rates[skip_start : -skip_end if skip_end > 0 else None])
        losses = np.array(self.losses[skip_start : -skip_end if skip_end > 0 else None])

        # Find steepest descent
        grad = np.gradient(losses)
        min_grad_idx = grad.argmin()

        suggested_lr = lrs[min_grad_idx]

        logger.info(f"Suggested learning rate: {suggested_lr:.2e}")

        return float(suggested_lr)


def find_lr(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = "cuda",
    start_lr: float = 1e-6,
    end_lr: float = 1e-2,
    num_iter: int = 200,
) -> float:
    """
    Convenience function to find optimal learning rate

    Args:
        model: Model
        train_loader: Training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device
        start_lr: Starting LR
        end_lr: Ending LR
        num_iter: Number of iterations

    Returns:
        Suggested learning rate
    """
    lr_finder = LRFinder(model, optimizer, criterion, device)

    lr_finder.range_test(train_loader, start_lr=start_lr, end_lr=end_lr, num_iter=num_iter)

    suggested_lr = lr_finder.suggest_lr()

    return suggested_lr


if __name__ == "__main__":
    # Test LR Finder
    print("LR Finder Test")
    print("=" * 60)

    # Create dummy model and data
    model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 2))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print(f"Using device: {device}")

    # Create dummy dataset
    from torch.utils.data import TensorDataset

    X = torch.randn(1000, 10)
    y = torch.randint(0, 2, (1000,))

    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"Created dummy dataset: {len(dataset)} samples")

    # Create optimizer and criterion
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Run LR finder
    print(f"\nRunning LR finder...")

    lr_finder = LRFinder(model, optimizer, criterion, device)

    lrs, losses = lr_finder.range_test(train_loader, start_lr=1e-6, end_lr=1e-2, num_iter=100)

    print(f"\nResults:")
    print(f"  LR range: [{min(lrs):.2e}, {max(lrs):.2e}]")
    print(f"  Loss range: [{min(losses):.4f}, {max(losses):.4f}]")

    # Get suggestion
    suggested_lr = lr_finder.suggest_lr()
    print(f"  Suggested LR: {suggested_lr:.2e}")

    # Test plot (don't show in automated test)
    # lr_finder.plot(save_path=Path("lr_finder_test.png"))

    print("\n✅ LR Finder test complete")
