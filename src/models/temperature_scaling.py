"""
Temperature Scaling for Model Calibration
Learns optimal temperature parameter to calibrate model confidence
"""
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class TemperatureScaling(nn.Module):
    """
    Temperature Scaling Module

    Learns a single scalar temperature parameter that scales logits
    to produce better-calibrated probabilities.

    Reference: "On Calibration of Modern Neural Networks" (Guo et al., 2017)
    """

    def __init__(self, initial_temperature: float = 1.0):
        """
        Initialize temperature scaling

        Args:
            initial_temperature: Initial temperature value
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits

        Args:
            logits: Model logits (batch, num_classes)

        Returns:
            Temperature-scaled logits
        """
        # Clamp temperature to avoid numerical instability
        temp = self.temperature.clamp(min=1e-3)
        return logits / temp

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
        verbose: bool = True,
    ) -> float:
        """
        Fit temperature parameter using validation set

        Args:
            logits: Validation set logits (N, num_classes)
            labels: Validation set labels (N,)
            lr: Learning rate for optimization
            max_iter: Maximum iterations
            verbose: Print optimization progress

        Returns:
            Final NLL loss value
        """
        # Move to same device as logits
        self.to(logits.device)

        # Use NLL loss for calibration
        criterion = nn.CrossEntropyLoss()

        # Optimize temperature
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss

        # Run optimization
        optimizer.step(closure)

        # Compute final loss
        with torch.no_grad():
            scaled_logits = self.forward(logits)
            final_loss = criterion(scaled_logits, labels).item()

        if verbose:
            logger.info(
                f"Temperature scaling fitted: "
                f"T={self.temperature.item():.4f}, "
                f"NLL={final_loss:.4f}"
            )

        return final_loss

    def get_temperature(self) -> float:
        """Get current temperature value"""
        return self.temperature.item()


def calibrate_model(
    model: nn.Module,
    val_loader,
    device: str = "cuda",
    lr: float = 0.01,
    max_iter: int = 50,
) -> TemperatureScaling:
    """
    Calibrate a trained model using temperature scaling

    Args:
        model: Trained model to calibrate
        val_loader: Validation data loader
        device: Device for computation
        lr: Learning rate
        max_iter: Max iterations

    Returns:
        Fitted TemperatureScaling module
    """
    logger.info("Calibrating model with temperature scaling...")

    model.eval()

    # Collect all logits and labels from validation set
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:
                inputs, labels, _ = batch
            else:
                inputs, labels = batch

            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all batches
    all_logits = torch.cat(all_logits, dim=0).to(device)
    all_labels = torch.cat(all_labels, dim=0).to(device)

    logger.info(f"Collected {len(all_logits)} validation samples")

    # Fit temperature scaling
    temp_scaling = TemperatureScaling()
    temp_scaling.fit(
        logits=all_logits, labels=all_labels, lr=lr, max_iter=max_iter, verbose=True
    )

    return temp_scaling


def apply_temperature_scaling(
    model: nn.Module, temp_scaling: TemperatureScaling
) -> nn.Module:
    """
    Wrap model with temperature scaling

    Args:
        model: Base model
        temp_scaling: Fitted temperature scaling module

    Returns:
        Wrapped model that applies temperature scaling
    """

    class CalibratedModel(nn.Module):
        def __init__(self, base_model, temp_module):
            super().__init__()
            self.base_model = base_model
            self.temp_scaling = temp_module

        def forward(self, x):
            logits = self.base_model(x)
            scaled_logits = self.temp_scaling(logits)
            return scaled_logits

    calibrated_model = CalibratedModel(model, temp_scaling)
    return calibrated_model


if __name__ == "__main__":
    # Test temperature scaling
    print("Temperature Scaling Test")
    print("=" * 60)

    # Create dummy logits and labels
    batch_size = 100
    num_classes = 2

    # Uncalibrated logits (overconfident)
    logits = torch.randn(batch_size, num_classes) * 5  # Large scale = overconfident
    labels = torch.randint(0, num_classes, (batch_size,))

    print(f"Test setup:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num classes: {num_classes}")

    # Check initial confidence
    probs = torch.softmax(logits, dim=1)
    max_probs = probs.max(dim=1)[0]
    print(f"\nBefore calibration:")
    print(f"  Mean confidence: {max_probs.mean():.4f}")
    print(f"  Median confidence: {max_probs.median():.4f}")

    # Fit temperature scaling
    temp_scaling = TemperatureScaling()
    print(f"\nFitting temperature scaling...")
    final_loss = temp_scaling.fit(logits, labels, verbose=True)

    print(f"  Fitted temperature: {temp_scaling.get_temperature():.4f}")
    print(f"  Final NLL: {final_loss:.4f}")

    # Check calibrated confidence
    scaled_logits = temp_scaling(logits)
    calibrated_probs = torch.softmax(scaled_logits, dim=1)
    calibrated_max_probs = calibrated_probs.max(dim=1)[0]

    print(f"\nAfter calibration:")
    print(f"  Mean confidence: {calibrated_max_probs.mean():.4f}")
    print(f"  Median confidence: {calibrated_max_probs.median():.4f}")

    # Verify temperature scaling preserves predictions
    original_preds = logits.argmax(dim=1)
    scaled_preds = scaled_logits.argmax(dim=1)

    assert torch.equal(
        original_preds, scaled_preds
    ), "Temperature scaling changed predictions!"
    print(f"\n✅ Predictions preserved after calibration")

    print("\n✅ Temperature scaling test complete")
