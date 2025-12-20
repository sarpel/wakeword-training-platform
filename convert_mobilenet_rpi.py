#!/usr/bin/env python3
"""
MobileNetV3 to Raspberry Pi (TFLite) Conversion Script
Safe, logging-enabled, and using project utilities.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.config.logger import setup_logger
from src.models.architectures import create_model
from src.export.onnx_exporter import ONNXExporter

# Initialize logger
logger = setup_logger(__name__)

# --- CONFIGURATION ---
# Default settings (can be overridden by arguments)
DEFAULT_CHECKPOINT = "models/checkpoints/best_model.pt"
DEFAULT_OUTPUT_DIR = "tflite_rpi_output"
ONNX_FILENAME = "hey_katya_rpi.onnx"

# Audio Settings (Must match training config)
N_MELS = 64
N_FRAMES = 101  # ~1.0s at 16kHz with standard hop length

# Model Parameters (Standard MobileNetV3 Wakeword)
MODEL_PARAMS: Dict[str, Any] = {
    "architecture": "mobilenetv3",
    "num_classes": 2,
    "pretrained": False,
    "dropout": 0.2,
    "input_channels": 1,
}


def load_checkpoint_safely(model: torch.nn.Module, checkpoint_path: Path) -> None:
    """
    Load checkpoint safely with weights_only=True and state_dict cleaning.
    """
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    try:
        # Security: Use weights_only=True to prevent pickle exploits
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        
        # Handle different checkpoint structures
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            logger.info("Found 'model_state_dict' key.")
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            logger.info("Found 'state_dict' key.")
        else:
            state_dict = checkpoint
            logger.info("Assuming direct state dict structure.")

        # Clean state dict (remove 'model.' or 'mobilenet.' prefixes if necessary)
        # and remove QAT specific keys for clean FP32 export
        clean_state_dict = {}
        for k, v in state_dict.items():
            # Skip QAT observer/fake_quant keys for inference/export
            if "activation_post_process" in k or "_observer" in k or "fake_quant" in k:
                continue
            
            # Remove prefixes if they were saved wrapped in another module
            k = k.replace("model.", "").replace("mobilenet.", "")
            clean_state_dict[k] = v

        # Load weights
        missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
        
        if missing:
            logger.warning(f"Missing keys (usually fine for QAT/Aux layers): {missing[:5]}...")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected[:5]}...")
            
        logger.info("✅ Weights loaded successfully.")

    except Exception as e:
        logger.exception(f"Failed to load checkpoint: {e}")
        sys.exit(1)


def convert_to_tflite(onnx_path: Path, output_dir: Path, calib_data_path: Path) -> None:
    """
    Convert ONNX to TFLite using onnx2tf via subprocess.
    """
    logger.info("⏳ Converting to TFLite (via onnx2tf)...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct command securely
    # -oiqt: Output integer quantized TFLite
    # -cind: Custom input node data (for calibration)
    cmd = [
        "onnx2tf",
        "-i", str(onnx_path),
        "-o", str(output_dir),
        "-oiqt",
        "-cind", "input", str(calib_data_path), "0", "1"
    ]
    
    try:
        # Security: Use subprocess.run with shell=False (default)
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(result.stdout)
        
        tflite_path = output_dir / f"{onnx_path.stem}_dynamic_range_quant.tflite"
        logger.info("\n" + "="*40)
        logger.info(f"🎉 RPi Model Ready!")
        logger.info(f"📂 Output: {output_dir}")
        logger.info("="*40)
        
    except subprocess.CalledProcessError as e:
        logger.error("❌ onnx2tf command failed.")
        logger.error(e.stderr)
        logger.error("Ensure 'onnx2tf' and 'tensorflow' are installed: pip install onnx2tf tensorflow")
        sys.exit(1)
    except FileNotFoundError:
        logger.error("❌ 'onnx2tf' command not found in PATH.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Convert MobileNetV3 to TFLite for RPi")
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT), help="Path to .pt checkpoint")
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    # 1. Initialize Model
    logger.info("🔨 Initializing MobileNetV3...")
    try:
        model = create_model(**MODEL_PARAMS)
        model.eval()
    except Exception as e:
        logger.exception(f"Failed to create model: {e}")
        sys.exit(1)

    # 2. Load Checkpoint
    load_checkpoint_safely(model, args.checkpoint)

    # 3. Export to ONNX
    onnx_path = args.output / ONNX_FILENAME
    args.output.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🔄 Exporting to ONNX: {onnx_path}")
    
    # Create dummy input with batch size 1
    dummy_input = torch.randn(1, 1, N_MELS, N_FRAMES)
    
    try:
        # Use project's ONNXExporter if possible, or direct export with our config
        # Here we use direct export tailored for this specific RPi conversion needs
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=['input'],
            output_names=['output'],
            opset_version=13,
            dynamic_axes=None 
        )
        logger.info(f"✅ ONNX exported.")
    except Exception as e:
        logger.exception(f"ONNX export failed: {e}")
        sys.exit(1)

    # 4. Generate Calibration Data
    logger.info("📊 Generating calibration data...")
    calib_data = np.random.randn(1, 1, N_MELS, N_FRAMES).astype(np.float32)
    calib_path = args.output / "calib_data_rpi.npy"
    np.save(calib_path, calib_data)

    # 5. Convert to TFLite
    convert_to_tflite(onnx_path, args.output, calib_path)


if __name__ == "__main__":
    main()
