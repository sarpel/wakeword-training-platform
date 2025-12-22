import pytest
import torch
import torch.nn.functional as F
from unittest.mock import MagicMock, patch
from src.training.distillation_trainer import DistillationTrainer
from src.config.defaults import WakewordConfig

class MockTeacher(torch.nn.Module):
    def __init__(self, output_val=1.0):
        super().__init__()
        self.output_val = output_val
        self.linear = torch.nn.Linear(10, 2)
    def forward(self, x):
        # Return constant logits for predictability
        batch_size = x.shape[0]
        return torch.ones(batch_size, 2) * self.output_val

@pytest.fixture
def dist_config():
    config = WakewordConfig()
    config.distillation.enabled = True
    config.distillation.alpha = 0.5
    config.distillation.temperature = 1.0
    config.model.num_classes = 2
    return config

def test_distillation_loss_computation(dist_config):
    """Test that distillation loss is computed correctly using KL Div."""
    
    # Mock Trainer dependencies
    model = MagicMock()
    model.to.return_value = model
    # Return fresh iterator each time
    param = torch.randn(1, requires_grad=True)
    model.parameters.side_effect = lambda: iter([param])
    
    # Init trainer
    with patch("src.training.distillation_trainer.Wav2VecWakeword") as MockWav2Vec:
        teacher_mock = MockTeacher(output_val=2.0)
        MockWav2Vec.return_value = teacher_mock
        
        trainer = DistillationTrainer(
            model=model,
            train_loader=MagicMock(),
            val_loader=MagicMock(),
            config=dist_config,
            checkpoint_manager=MagicMock(),
            device="cpu"
        )
        
        # Manually set teacher to ensure it's our simple mock
        trainer.teacher = teacher_mock
        trainer.teacher_device = torch.device("cpu")
        
        # Inputs
        batch_size = 2
        outputs = torch.zeros(batch_size, 2) # Student logits (0,0)
        targets = torch.zeros(batch_size).long() # Labels
        inputs = torch.randn(batch_size, 16000) # Raw audio
        
        # Student loss (CrossEntropy)
        # CE(logits=[0,0], target=0) -> -log(0.5) = 0.693
        # We need to mock super().compute_loss call or calculate it manually if we can't easily mock super
        
        # Since DistillationTrainer inherits from Trainer, and we want to test compute_loss logic:
        # We can mock the super().compute_loss method if possible, OR just rely on Trainer's implementation (which uses criterion)
        
        # Let's set the criterion to standard CE
        trainer.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
        loss = trainer.compute_loss(outputs, targets, inputs=inputs)
        
        # Expected calculation:
        # Student Loss: 0.6931
        # Teacher Logits: [2.0, 2.0]
        # Student Logits: [0.0, 0.0]
        # T = 1.0
        
        # KL(Student || Teacher)
        # Soft Student: softmax([0,0]) = [0.5, 0.5]
        # Soft Teacher: softmax([2,2]) = [0.5, 0.5]
        # KL([0.5, 0.5] || [0.5, 0.5]) = 0.0
        
        # Total Loss = 0.5 * 0.6931 + 0.5 * 0.0 = 0.3465
        
        assert torch.isclose(loss, torch.tensor(0.3465), atol=1e-4)

def test_distillation_loss_computation_diff_distribution(dist_config):
    """Test with different distributions to ensure KL is non-zero."""
    
    # Mock Trainer dependencies
    model = MagicMock()
    model.to.return_value = model
    param = torch.randn(1, requires_grad=True)
    model.parameters.side_effect = lambda: iter([param])
    
    with patch("src.training.distillation_trainer.Wav2VecWakeword"):
        trainer = DistillationTrainer(
            model=model,
            train_loader=MagicMock(),
            val_loader=MagicMock(),
            config=dist_config,
            checkpoint_manager=MagicMock(),
            device="cpu"
        )
        
        # Set Teacher to output logits favoring class 1 significantly
        # Teacher Logits: [0, 10] -> Prob ~ [0, 1]
        trainer.teacher = MagicMock()
        trainer.teacher.return_value = torch.tensor([[0.0, 10.0]])
        trainer.teacher_device = torch.device("cpu")
        
        # Student Logits: [10, 0] -> Prob ~ [1, 0] (Opposite)
        outputs = torch.tensor([[10.0, 0.0]])
        targets = torch.tensor([1]) # Target class 1
        inputs = torch.randn(1, 16000)
        
        trainer.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
        loss = trainer.compute_loss(outputs, targets, inputs=inputs)
        
        # Student Loss: CE([10, 0], 1) -> Large loss (since it predicts 0)
        # KL(Student || Teacher): Large (distributions are opposite)
        
        # We just check it runs and produces a valid scalar
        assert loss.dim() == 0
        assert loss.item() > 0

def test_teacher_loading_security(dist_config):
    """Test that teacher loading enforces directory constraints."""
    
    model = MagicMock()
    model.to.return_value = model
    param = torch.randn(1, requires_grad=True)
    model.parameters.side_effect = lambda: iter([param])

    with patch("src.training.distillation_trainer.Wav2VecWakeword"):
        trainer = DistillationTrainer(
            model=model,
            train_loader=MagicMock(),
            val_loader=MagicMock(),
            config=dist_config,
            checkpoint_manager=MagicMock(),
            device="cpu"
        )
        
        # Test loading from restricted path
        with pytest.raises(ValueError, match="must be in allowed directories"):
            trainer._load_teacher_checkpoint("/tmp/malicious_checkpoint.pt")
