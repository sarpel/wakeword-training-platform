"""
Tests for model size validation warnings during export.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
from src.export.onnx_exporter import export_model_to_onnx
from src.config.defaults import get_default_config, SizeTargetConfig

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        # Flatten input to (batch, 10) for the linear layer
        # Input shape is (batch, 1, n_mels, n_frames)
        x = x.view(x.size(0), -1)
        # If input size doesn't match, we might need to adjust, 
        # but for mocking it's fine as long as we don't run it deeply.
        return self.fc(x[:, :10])

class TestExportSizeWarning:
    """Test size warnings during export"""

    @patch("torch.load")
    @patch("src.models.architectures.create_model")
    @patch("src.export.onnx_exporter.ONNXExporter")
    def test_size_warning_trigger(self, mock_exporter_cls, mock_create_model, mock_torch_load, tmp_path):
        """Test that size warnings are correctly identified when targets are exceeded"""
        
        # 1. Setup mock config with strict size targets
        config = get_default_config()
        config.size_targets = SizeTargetConfig(max_flash_kb=10) # 10KB is very small
        
        # 2. Setup mock checkpoint
        mock_checkpoint = {
            "config": config,
            "model_state_dict": {
                "fc.weight": torch.randn(2, 10),
                "fc.bias": torch.randn(2)
            }
        }
        mock_torch_load.return_value = mock_checkpoint
        
        # 3. Setup mock model
        mock_model = TinyModel()
        mock_create_model.return_value = mock_model
        
        # 4. Setup mock exporter instance
        mock_exporter = MagicMock()
        mock_exporter_cls.return_value = mock_exporter
        
        # Mock results showing a 50KB model (exceeds 10KB target)
        mock_exporter.export.return_value = {
            "success": True,
            "path": str(tmp_path / "model.onnx"),
            "file_size_mb": 0.05 # 50KB
        }
        
        # 5. Run export
        output_path = tmp_path / "test.onnx"
        results = export_model_to_onnx(
            checkpoint_path=Path("dummy.pt"),
            output_path=output_path,
            device="cpu"
        )
        
        # 6. Verify warning
        assert results["success"] is True
        assert results["size_warning"] is True
        assert results["file_size_mb"] == 0.05

    @patch("torch.load")
    @patch("src.models.architectures.create_model")
    @patch("src.export.onnx_exporter.ONNXExporter")
    def test_no_warning_when_within_target(self, mock_exporter_cls, mock_create_model, mock_torch_load, tmp_path):
        """Test that no warning is issued when under target"""
        config = get_default_config()
        config.size_targets = SizeTargetConfig(max_flash_kb=100) # 100KB target
        
        mock_torch_load.return_value = {
            "config": config, 
            "model_state_dict": {
                "fc.weight": torch.randn(2, 10),
                "fc.bias": torch.randn(2)
            }
        }
        mock_create_model.return_value = TinyModel()
        
        mock_exporter = MagicMock()
        mock_exporter_cls.return_value = mock_exporter
        mock_exporter.export.return_value = {
            "success": True,
            "path": str(tmp_path / "model.onnx"),
            "file_size_mb": 0.05 # 50KB (under 100KB)
        }
        
        results = export_model_to_onnx(Path("dummy.pt"), tmp_path / "test.onnx", device="cpu")
        
        assert results["size_warning"] is False
