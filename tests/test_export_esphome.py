"""
Integration tests for ESPHome export compatibility
"""
import pytest
from pathlib import Path
import torch
import torch.nn as nn
from src.export.onnx_exporter import export_model_to_onnx
from src.config.defaults import WakewordConfig

def test_esphome_fixed_path_export(tmp_path):
    """Test that model is copied to the fixed ESPHome path"""
    # Create a dummy checkpoint
    checkpoint_path = tmp_path / "dummy.pt"
    config = WakewordConfig()
    config.model.architecture = "tiny_conv"
    
    # Simple dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 16, 3)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, 2)
        def forward(self, x):
            return self.fc(self.pool(self.conv(x)).flatten(1))
            
    model = DummyModel()
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict()
    }, checkpoint_path)
    
    output_path = tmp_path / "test.onnx"
    fixed_path = tmp_path / "esphome" / "wakeword.tflite"
    
    # We mock tflite export success because onnx2tf might not be installed in CI
    # But we want to test the copying logic in export_model_to_onnx
    
    # Wait, export_model_to_onnx calls exporter.export_to_tflite
    # If onnx2tf is missing, it will fail.
    
    # Let's test the underlying logic if possible, or just skip if onnx2tf missing
    import subprocess
    try:
        subprocess.run(["onnx2tf", "--version"], capture_output=True, check=True)
        HAS_ONNX2TF = True
    except:
        HAS_ONNX2TF = False
        
    if not HAS_ONNX2TF:
        pytest.skip("onnx2tf not installed, skipping integration test")

    results = export_model_to_onnx(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        export_tflite=True,
        quantize_int8=True,
        fixed_export_path=fixed_path,
        device="cpu"
    )
    
    assert results["success"] is True
    assert fixed_path.exists()
    assert results["fixed_path"] == str(fixed_path)
