"""
Advanced Unit Tests for Focal Loss Implementation.
Focuses on mathematical correctness, gradient flow, and parameter sensitivity.
"""

import pytest
import torch
import torch.nn.functional as F
from src.models.losses import FocalLoss

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestFocalLossAdvanced:
    """Advanced test suite for FocalLoss."""

    @pytest.mark.unit
    def test_gradient_flow(self, device):
        """Verify that gradients are non-zero and flow back to inputs."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0).to(device)
        
        logits = torch.randn(4, 2, device=device, requires_grad=True)
        targets = torch.tensor([0, 1, 0, 1], device=device)
        
        loss = loss_fn(logits, targets)
        loss.backward()
        
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)
        assert not torch.isnan(logits.grad).any()

    @pytest.mark.unit
    def test_gamma_sensitivity(self, device):
        """Verify that higher gamma leads to lower loss for easy examples."""
        # Moderate example: logit of 2.0 for correct class, -2.0 for incorrect
        # p_t will be approx exp(2)/(exp(2)+exp(-2)) = 7.389 / (7.389 + 0.135) = 0.982
        easy_logits = torch.tensor([[2.0, -2.0]], device=device, requires_grad=True)
        target = torch.tensor([0], device=device)
        
        loss_g0 = FocalLoss(alpha=None, gamma=0.0).to(device) # Cross Entropy equivalent
        loss_g2 = FocalLoss(alpha=None, gamma=2.0).to(device)
        loss_g5 = FocalLoss(alpha=None, gamma=5.0).to(device)
        
        val_g0 = loss_g0(easy_logits, target).item()
        val_g2 = loss_g2(easy_logits, target).item()
        val_g5 = loss_g5(easy_logits, target).item()
        
        print(f"Easy example losses - Gamma 0: {val_g0:.8f}, Gamma 2: {val_g2:.8f}, Gamma 5: {val_g5:.8f}")
        
        assert val_g2 < val_g0
        assert val_g5 < val_g2

    @pytest.mark.unit
    def test_alpha_weighting(self, device):
        """Verify that alpha correctly weights positive vs negative classes."""
        # Case where alpha=0.25 (weighting positive class 1 as 0.25 and negative class 0 as 0.75)
        # Note: In the implementation, target==1 gets alpha, target==0 gets 1-alpha.
        loss_fn = FocalLoss(alpha=0.25, gamma=0.0, reduction='none').to(device)
        
        logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]], device=device) # Neutral predictions
        targets = torch.tensor([0, 1], device=device)
        
        loss = loss_fn(logits, targets)
        
        # Cross entropy for neutral prediction is -log(0.5) approx 0.6931
        # Target 0 (negative) should be multiplied by (1 - 0.25) = 0.75
        # Target 1 (positive) should be multiplied by 0.25
        
        expected_neg = 0.6931 * 0.75
        expected_pos = 0.6931 * 0.25
        
        assert torch.allclose(loss[0], torch.tensor(expected_neg, device=device), atol=1e-3)
        assert torch.allclose(loss[1], torch.tensor(expected_pos, device=device), atol=1e-3)

    @pytest.mark.unit
    def test_multi_class_compatibility(self, device):
        """Verify FocalLoss works with more than 2 classes."""
        num_classes = 5
        loss_fn = FocalLoss(alpha=None, gamma=2.0).to(device)
        
        logits = torch.randn(8, num_classes, device=device)
        targets = torch.randint(0, num_classes, (8,), device=device)
        
        loss = loss_fn(logits, targets)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    @pytest.mark.unit
    def test_reduction_modes(self, device):
        """Verify 'mean', 'sum', and 'none' reductions."""
        logits = torch.randn(4, 2, device=device)
        targets = torch.tensor([0, 1, 0, 1], device=device)
        
        l_none = FocalLoss(reduction='none').to(device)(logits, targets)
        l_mean = FocalLoss(reduction='mean').to(device)(logits, targets)
        l_sum = FocalLoss(reduction='sum').to(device)(logits, targets)
        
        assert l_none.shape == (4,)
        assert torch.allclose(l_mean, l_none.mean())
        assert torch.allclose(l_sum, l_none.sum())

if __name__ == "__main__":
    # Manual run if needed
    pytest.main([__file__])
