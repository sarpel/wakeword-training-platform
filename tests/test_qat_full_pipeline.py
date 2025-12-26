"""
Full Pipeline Integration Test: Standard Training -> QAT -> ONNX/TFLite Export.
"""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config.defaults import WakewordConfig
from src.export.onnx_exporter import export_model_to_onnx
from src.models.architectures import create_model
from src.training.checkpoint_manager import CheckpointManager
from src.training.trainer import Trainer


@pytest.mark.integration
def test_full_pipeline_qat_export(tmp_path):
    device = "cpu"  # Use CPU for CI reliability

    # 1. Setup Config
    config = WakewordConfig()
    config.training.epochs = 2
    config.training.save_best_only = True
    config.qat.enabled = True
    config.qat.start_epoch = 1
    config.model.architecture = "tiny_conv"
    # Ensure structural parameters are explicitly set to match creation
    config.model.tcn_num_channels = [16, 32, 64, 64]

    # 2. Setup Data
    dummy_data = torch.randn(20, 1, 64, 50)
    dummy_labels = torch.randint(0, 2, (20,))
    dataset = TensorDataset(dummy_data, dummy_labels)
    loader = DataLoader(dataset, batch_size=4)

    # 3. Train
    model = create_model("tiny_conv", num_classes=2)
    checkpoint_manager = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")
    trainer = Trainer(model, loader, loader, config, checkpoint_manager, device=device)

    results = trainer.train()
    assert results is not None
    assert "qat_report" in results

    # 4. Export to ONNX
    onnx_path = tmp_path / "model.onnx"
    checkpoint_path = trainer.checkpoint_manager.checkpoint_dir / "best_model.pt"

    # We will try to export to TFLite but skip if onnx2tf is missing or fails
    try:
        export_results = export_model_to_onnx(
            checkpoint_path=checkpoint_path, output_path=onnx_path, export_tflite=True, device=device
        )

        assert export_results["success"]
        assert onnx_path.exists()

        # If TFLite export was attempted, check it
        if export_results.get("tflite_success"):
            assert Path(str(onnx_path).replace(".onnx", ".tflite")).exists()
            print("✅ TFLite export successful in integration test")

    except Exception as e:
        pytest.fail(f"Export failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
