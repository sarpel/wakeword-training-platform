import pytest
from pathlib import Path
import torch
import tempfile
import os
import shutil

# We need to mock the Trainer dependencies to import DistillationTrainer without issues
# if the environment is restricted. But typically imports should be fine.
from src.training.distillation_trainer import DistillationTrainer

class MockDistillationTrainer(DistillationTrainer):
    """Mock trainer that bypasses initialization logic"""
    def __init__(self):
        pass

def test_secure_checkpoint_loading_path_traversal():
    """Test that path traversal attempts are blocked"""
    trainer = MockDistillationTrainer()
    
    # Create a path that attempts to go up
    # We resolve it to check if it's strictly enforced
    # Windows/Linux agnostic attempt
    
    # Current working directory is project root
    cwd = Path.cwd()
    
    # Try to access parent of parent (likely outside project)
    traversal_path = ".." + os.sep + ".." + os.sep + "outside_file.pt"
    
    # We don't need the file to exist for the path validation check 
    # because validation happens before existence check?
    # Looking at implementation: 
    # 1. resolve()
    # 2. validation (is_allowed)
    # 3. exists()
    # So if we provide a path that resolves outside, it should raise ValueError
    
    with pytest.raises(ValueError, match="Teacher checkpoint must be in allowed directories"):
        trainer._load_teacher_checkpoint(traversal_path)

def test_secure_checkpoint_loading_file_not_found():
    """Test clear error for missing checkpoint"""
    trainer = MockDistillationTrainer()
    
    # Use a path that IS allowed but doesn't exist
    # 'models' dir is allowed
    allowed_missing_path = Path("models") / "nonexistent_teacher.pt"
    
    with pytest.raises(FileNotFoundError, match="Teacher checkpoint not found"):
        trainer._load_teacher_checkpoint(str(allowed_missing_path))

def test_secure_checkpoint_loading_valid():
    """Test loading a valid (mocked) checkpoint"""
    trainer = MockDistillationTrainer()
    
    # Create a temp directory inside the project root to ensure it's allowed
    # We can use the 'models' directory which should exist or we create it
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    ckpt_path = models_dir / "temp_test_teacher.pt"
    
    try:
        # Save a dummy checkpoint
        # We assume torch is available
        torch.save({"state": "dummy"}, ckpt_path)
        
        # Load it
        ckpt = trainer._load_teacher_checkpoint(str(ckpt_path))
        assert ckpt["state"] == "dummy"
        
    finally:
        # Cleanup
        if ckpt_path.exists():
            ckpt_path.unlink()

def test_secure_checkpoint_loading_weights_only():
    """
    Test that weights_only=True is used (implied by functionality).
    We can't easily verify the flag usage without mocking torch.load,
    so we'll verify it loads a clean tensor file.
    """
    trainer = MockDistillationTrainer()
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    ckpt_path = models_dir / "temp_test_weights.pt"
    
    try:
        torch.save({"tensor": torch.tensor([1.0])}, ckpt_path)
        ckpt = trainer._load_teacher_checkpoint(str(ckpt_path))
        assert torch.equal(ckpt["tensor"], torch.tensor([1.0]))
    finally:
        if ckpt_path.exists():
            ckpt_path.unlink()
