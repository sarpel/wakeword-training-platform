"""
Loss Functions for Wakeword Detection
Includes: Cross Entropy with Label Smoothing, Focal Loss
"""
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing
    Prevents overconfidence and improves generalization
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        """
        Initialize Label Smoothing Cross Entropy Loss

        Args:
            smoothing: Label smoothing factor (0.0-1.0)
            weight: Class weights tensor
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        self.confidence = 1.0 - smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            pred: Model predictions (logits) (batch, num_classes)
            target: Ground truth labels (batch,)

        Returns:
            Loss value
        """
        # Get log probabilities
        log_probs = F.log_softmax(pred, dim=-1)

        # One-hot encode targets
        num_classes = pred.size(-1)
        target_one_hot = F.one_hot(target, num_classes).float()

        # Apply label smoothing
        smooth_target = target_one_hot * self.confidence + (1 - target_one_hot) * (
            self.smoothing / (num_classes - 1)
        )

        # Calculate loss
        loss = -torch.sum(smooth_target * log_probs, dim=-1)

        # Apply class weights if provided
        if self.weight is not None:
            weight = self.weight[target]
            loss = loss * weight

        # Apply reduction
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Focuses on hard examples by down-weighting easy examples
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        """
        Initialize Focal Loss

        Args:
            alpha: Weighting factor for class 1 (0.0-1.0), use None if weight is provided
            gamma: Focusing parameter (higher = more focus on hard examples)
            weight: Class weights tensor, use None if alpha is provided
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super().__init__()

        # Guard: Don't use both alpha and class_weights simultaneously
        if alpha is not None and weight is not None:
            logger.warning(
                "Both alpha and class_weights provided. Using class_weights only, ignoring alpha."
            )
            self.alpha = None
            self.weight = weight
        else:
            self.alpha = alpha
            self.weight = weight

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            pred: Model predictions (logits) (batch, num_classes)
            target: Ground truth labels (batch,)

        Returns:
            Loss value
        """
        # Get probabilities
        probs = F.softmax(pred, dim=-1)

        # Get class probabilities
        target_one_hot = F.one_hot(target, pred.size(-1)).float()
        pt = torch.sum(probs * target_one_hot, dim=-1)

        # Calculate focal term: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # Calculate cross entropy
        ce_loss = F.cross_entropy(pred, target, reduction="none")

        # Apply focal weight
        loss = focal_weight * ce_loss

        # Calculate alpha term (only if no class_weights provided)
        if self.alpha is not None and self.weight is None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = torch.where(
                    target == 1,
                    torch.tensor(self.alpha, device=target.device),
                    torch.tensor(1 - self.alpha, device=target.device),
                )
            else:
                # Alpha is a tensor of class weights
                alpha_t = self.alpha[target]
            loss = alpha_t * loss

        # Apply class weights if provided (mutually exclusive with alpha)
        if self.weight is not None:
            weight = self.weight[target]
            loss = loss * weight

        # Apply reduction
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


def create_loss_function(
    loss_name: str,
    num_classes: int = 2,
    label_smoothing: float = 0.1,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    class_weights: Optional[torch.Tensor] = None,
    device: str = "cuda",
) -> nn.Module:
    """
    Factory function to create loss functions

    Args:
        loss_name: Loss function name ('cross_entropy', 'focal_loss')
        num_classes: Number of classes
        label_smoothing: Label smoothing factor for cross entropy
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
        class_weights: Optional class weights tensor
        device: Device to place loss on

    Returns:
        Loss function module

    Raises:
        ValueError: If loss_name is not recognized
    """
    loss_name = loss_name.lower()

    # Move class weights to device if provided
    if class_weights is not None:
        class_weights = class_weights.to(device)

    if loss_name == "cross_entropy":
        if label_smoothing > 0:
            return LabelSmoothingCrossEntropy(
                smoothing=label_smoothing, weight=class_weights, reduction="mean"
            )
        else:
            return nn.CrossEntropyLoss(weight=class_weights, reduction="mean")

    elif loss_name == "focal_loss":
        return FocalLoss(
            alpha=focal_alpha, gamma=focal_gamma, weight=class_weights, reduction="mean"
        )

    else:
        raise ValueError(
            f"Unknown loss function: {loss_name}. "
            f"Supported: cross_entropy, focal_loss"
        )


if __name__ == "__main__":
    # Test loss functions
    print("Loss Functions Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create dummy data
    batch_size = 4
    num_classes = 2
    pred = torch.randn(batch_size, num_classes).to(device)
    target = torch.randint(0, num_classes, (batch_size,)).to(device)

    print(f"\nTest input:")
    print(f"  Predictions shape: {pred.shape}")
    print(f"  Targets shape: {target.shape}")

    # Test Cross Entropy with Label Smoothing
    print("\n1. Testing Label Smoothing Cross Entropy...")
    ce_loss = LabelSmoothingCrossEntropy(smoothing=0.1)
    ce_loss = ce_loss.to(device)
    loss_value = ce_loss(pred, target)
    print(f"  Loss value: {loss_value.item():.4f}")
    print(f"  ✅ Label Smoothing Cross Entropy works")

    # Test Focal Loss
    print("\n2. Testing Focal Loss...")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    focal_loss = focal_loss.to(device)
    loss_value = focal_loss(pred, target)
    print(f"  Loss value: {loss_value.item():.4f}")
    print(f"  ✅ Focal Loss works")

    # Test with class weights
    print("\n3. Testing with class weights...")
    class_weights = torch.tensor([0.3, 0.7]).to(device)
    ce_weighted = LabelSmoothingCrossEntropy(smoothing=0.1, weight=class_weights)
    ce_weighted = ce_weighted.to(device)
    loss_value = ce_weighted(pred, target)
    print(f"  Loss value: {loss_value.item():.4f}")
    print(f"  ✅ Weighted loss works")

    # Test factory function
    print("\n4. Testing factory function...")
    loss_fn = create_loss_function("cross_entropy", label_smoothing=0.1, device=device)
    loss_value = loss_fn(pred, target)
    print(f"  Loss value: {loss_value.item():.4f}")
    print(f"  ✅ Factory function works")

    print("\n✅ All loss functions tested successfully")
    print("Loss functions module loaded successfully")
