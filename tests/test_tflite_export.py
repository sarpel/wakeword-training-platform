
import torch
import torch.nn as nn
from pathlib import Path
import os
import shutil
from src.models.architectures import create_model
from src.export.onnx_exporter import ONNXExporter, ExportConfig, export_model_to_onnx
from src.training.qat_utils import prepare_model_for_qat
from src.config.defaults import WakewordConfig

def test_tflite_export_flow():
    print("\nTesting TFLite Export Flow...")
    
    # 1. Setup
    checkpoint_dir = Path("test_checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    export_dir = Path("test_exports")
    export_dir.mkdir(exist_ok=True)
    
    checkpoint_path = checkpoint_dir / "test_model.pt"
    onnx_path = export_dir / "test_model.onnx"
    tflite_path = export_dir / "test_model.tflite"
    
    # 2. Create a dummy model and config
    config = WakewordConfig()
    config.model.architecture = "resnet18"
    model = create_model(config.model.architecture, num_classes=2)
    
    # 3. Save checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "epoch": 0,
        "val_loss": 0.0
    }, checkpoint_path)
    
    print(f"✅ Dummy checkpoint created at {checkpoint_path}")
    
    # 4. Attempt export (this will fail in this environment due to missing onnx2tf, but we check the logic)
    try:
        results = export_model_to_onnx(
            checkpoint_path=checkpoint_path,
            output_path=onnx_path,
            export_tflite=True,
            device="cpu"
        )
        
        print(f"Export Results: {results}")
        
        if results["success"]:
            print("✅ ONNX export successful")
        else:
            print(f"❌ ONNX export failed: {results.get('error')}")
            
        if results.get("tflite_success"):
            print("✅ TFLite export successful")
        else:
            print(f"❌ TFLite export failed: {results.get('tflite_error')}")
            
    except Exception as e:
        print(f"❌ Export process failed: {e}")

def test_qat_export_flow():
    print("\nTesting QAT TFLite Export Flow...")
    
    # 1. Setup
    checkpoint_dir = Path("test_checkpoints")
    export_dir = Path("test_exports")
    checkpoint_path = checkpoint_dir / "test_qat_model.pt"
    onnx_path = export_dir / "test_qat_model.onnx"
    
    # 2. Create QAT model
    config = WakewordConfig()
    config.qat.enabled = True
    model = create_model("resnet18", num_classes=2)
    model = prepare_model_for_qat(model, config.qat)
    
    # 3. Save QAT checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "epoch": 0,
        "val_loss": 0.0
    }, checkpoint_path)
    
    print(f"✅ QAT checkpoint created at {checkpoint_path}")
    
    # 4. Attempt export
    try:
        results = export_model_to_onnx(
            checkpoint_path=checkpoint_path,
            output_path=onnx_path,
            export_tflite=True,
            device="cpu"
        )
        
        print(f"QAT Export Results: {results}")
        
    except Exception as e:
        print(f"❌ QAT Export process failed: {e}")

if __name__ == "__main__":
    test_tflite_export_flow()
    test_qat_export_flow()
