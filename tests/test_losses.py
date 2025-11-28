"""
Unit Tests for Loss Functions
Tests all loss function implementations
"""
import pytest
import torch
import torch.nn as nn


class TestLossFunctions:
    """Test suite for custom loss functions"""

    @pytest.mark.unit
    def test_create_cross_entropy_loss(self, device):
        """Test CrossEntropy loss creation and computation"""
        from src.models.losses import create_loss_function
        
        loss_fn = create_loss_function("cross_entropy", num_classes=2, device=device)
        
        predictions = torch.randn(4, 2).to(device)
        targets = torch.tensor([0, 1, 0, 1]).to(device)
        
        loss = loss_fn(predictions, targets)
        
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"

    @pytest.mark.unit
    def test_create_focal_loss(self, device):
        """Test Focal Loss creation and computation"""
        from src.models.losses import create_loss_function
        
        loss_fn = create_loss_function(
            "focal",
            num_classes=2,
            focal_alpha=0.25,
            focal_gamma=2.0,
            device=device
        )
        
        predictions = torch.randn(4, 2).to(device)
        targets = torch.tensor([0, 1, 0, 1]).to(device)
        
        loss = loss_fn(predictions, targets)
        
        assert loss.dim() == 0, "Loss should be scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"

    @pytest.mark.unit
    def test_label_smoothing_loss(self, device):
        """Test Label Smoothing CrossEntropy"""
        from src.models.losses import create_loss_function
        
        loss_fn = create_loss_function(
            "cross_entropy",
            num_classes=2,
            label_smoothing=0.1,
            device=device
        )
        
        predictions = torch.randn(4, 2).to(device)
        targets = torch.tensor([0, 1, 0, 1]).to(device)
        
        loss = loss_fn(predictions, targets)
        
        assert not torch.isnan(loss), "Loss should not be NaN"

    @pytest.mark.unit
    def test_focal_loss_reduces_easy_examples(self, device):
        """Test Focal Loss down-weights easy examples"""
        from src.models.losses import create_loss_function
        
        focal_loss = create_loss_function(
            "focal",
            num_classes=2,
            focal_gamma=2.0,
            device=device
        )
        ce_loss = create_loss_function("cross_entropy", num_classes=2, device=device)
        
        # Easy example: high confidence correct prediction
        easy_pred = torch.tensor([[5.0, -5.0], [5.0, -5.0]]).to(device)  # Confident class 0
        easy_target = torch.tensor([0, 0]).to(device)
        
        focal_easy = focal_loss(easy_pred, easy_target)
        ce_easy = ce_loss(easy_pred, easy_target)
        
        # Focal loss should be lower for easy examples
        assert focal_easy.item() <= ce_easy.item()

    @pytest.mark.unit
    def test_loss_gradient_flow(self, device):
        """Test gradients flow through loss"""
        from src.models.losses import create_loss_function
        
        loss_fn = create_loss_function("cross_entropy", num_classes=2, device=device)
        
        predictions = torch.randn(4, 2, requires_grad=True).to(device)
        targets = torch.tensor([0, 1, 0, 1]).to(device)
        
        loss = loss_fn(predictions, targets)
        loss.backward()
        
        assert predictions.grad is not None, "Gradients should exist"
        assert not torch.all(predictions.grad == 0), "Gradients should be non-zero"


class TestTripletLoss:
    """Test Triplet Loss for metric learning"""

    @pytest.mark.unit
    def test_triplet_loss_creation(self, device):
        """Test TripletLoss can be created"""
        from src.models.losses import TripletLoss
        
        loss_fn = TripletLoss(margin=0.3)
        assert loss_fn is not None

    @pytest.mark.unit
    def test_triplet_loss_computation(self, device):
        """Test TripletLoss computation with embeddings"""
        from src.models.losses import TripletLoss
        
        loss_fn = TripletLoss(margin=0.3)
        
        # Batch of embeddings
        embeddings = torch.randn(8, 128).to(device)
        labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1]).to(device)
        
        loss = loss_fn(embeddings, labels)
        
        assert loss.dim() == 0, "Loss should be scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"
