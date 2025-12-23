"""
Panel 5: ONNX Export
- Convert PyTorch models to ONNX format
- FP16 and INT8 quantization
- Model validation and benchmarking
- Performance comparison
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
import structlog
import torch

from src.export.onnx_exporter import benchmark_onnx_model, export_model_to_onnx, validate_onnx_model

logger = structlog.get_logger(__name__)


class ExportState:
    """Global export state manager"""

    def __init__(self) -> None:
        self.last_export_path: Optional[Path] = None
        self.last_checkpoint: Optional[Path] = None
        self.export_results: Dict = {}
        self.validation_results: Dict = {}
        self.benchmark_results: Dict = {}


# Global state
export_state = ExportState()


def get_available_checkpoints() -> List[str]:
    """
    Get list of available model checkpoints

    Returns:
        List of checkpoint paths
    """
    checkpoint_dir = Path("models/checkpoints")

    if not checkpoint_dir.exists():
        return ["No checkpoints available"]

    # Find all checkpoint files
    checkpoints = list(checkpoint_dir.glob("*.pt"))

    if not checkpoints:
        return ["No checkpoints available"]

    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return [str(p) for p in checkpoints]


def export_to_onnx(
    checkpoint_path: str,
    output_filename: str,
    opset_version: int,
    dynamic_batch: bool,
    quantize_fp16: bool,
    quantize_int8: bool,
    export_tflite: bool = False,
    esphome_compatible: bool = False,
) -> Tuple[str, str]:
    """
    Export PyTorch model to ONNX

    Args:
        checkpoint_path: Path to checkpoint
        output_filename: Output filename
        opset_version: ONNX opset version
        dynamic_batch: Enable dynamic batch size
        quantize_fp16: Apply FP16 quantization
        quantize_int8: Apply INT8 quantization
        export_tflite: Export to TFLite (via onnx2tf)
        esphome_compatible: Copy to fixed path for ESPHome

    Returns:
        Tuple of (status_message, log_message)
    """
    if checkpoint_path == "No checkpoints available":
        return "❌ No checkpoints available. Train a model first (Panel 3).", ""

    if not checkpoint_path or checkpoint_path.strip() == "":
        return "❌ Please select a checkpoint", ""

    if not output_filename or output_filename.strip() == "":
        return "❌ Please provide an output filename", ""

    try:
        checkpoint_path_obj = Path(checkpoint_path)

        if not checkpoint_path_obj.exists():
            return f"❌ Checkpoint not found: {checkpoint_path_obj}", ""

        # Create output path
        export_dir = Path("models/exports")
        export_dir.mkdir(parents=True, exist_ok=True)

        output_path = export_dir / output_filename

        # ESPHome fixed path
        fixed_path = Path("exports/esphome/wakeword.tflite") if esphome_compatible else None

        logger.info(f"Exporting {checkpoint_path_obj} to {output_path}")

        # Build log message
        log = f"[{time.strftime('%H:%M:%S')}] Starting ONNX export...\n"
        log += f"Checkpoint: {checkpoint_path_obj.name}\n"
        log += f"Output: {output_filename}\n"
        log += f"Opset version: {opset_version}\n"
        log += f"Dynamic batch: {dynamic_batch}\n"
        log += f"FP16 quantization: {quantize_fp16}\n"
        log += f"INT8 quantization: {quantize_int8 or esphome_compatible}\n"
        log += f"TFLite Export: {export_tflite or esphome_compatible}\n"
        if esphome_compatible:
            log += f"ESPHome Compatible: YES (Target: {fixed_path})\n"
        log += "-" * 60 + "\n"

        # Export
        results = export_model_to_onnx(
            checkpoint_path=checkpoint_path_obj,
            output_path=output_path,
            opset_version=opset_version,
            dynamic_batch=dynamic_batch,
            quantize_fp16=quantize_fp16,
            quantize_int8=quantize_int8 or esphome_compatible,
            export_tflite=export_tflite or esphome_compatible,
            device="cuda",
            fixed_export_path=fixed_path,
        )

        if not results.get("success", False):
            error_msg = results.get("error", "Unknown error")
            log += f"❌ Export failed: {error_msg}\n"
            return f"❌ Export failed: {error_msg}", log

        # Update state
        export_state.last_export_path = output_path
        export_state.last_checkpoint = checkpoint_path_obj
        export_state.export_results = results

        # Build success message
        log += f"✅ Base model exported successfully\n"
        log += f"   File size: {results['file_size_mb']:.2f} MB\n"
        log += f"   Path: {output_path}\n"

        if quantize_fp16 and "fp16_path" in results:
            log += f"\n✅ FP16 model exported\n"
            log += f"   File size: {results['fp16_size_mb']:.2f} MB\n"
            log += f"   Reduction: {results['fp16_reduction']:.1f}%\n"
            log += f"   Path: {results['fp16_path']}\n"

        if quantize_int8 and "int8_path" in results:
            log += f"\n✅ INT8 model exported\n"
            log += f"   File size: {results['int8_size_mb']:.2f} MB\n"
            log += f"   Reduction: {results['int8_reduction']:.1f}%\n"
            log += f"   Path: {results['int8_path']}\n"

        if export_tflite and results.get("tflite_success", False):
            log += f"\n✅ TFLite model exported\n"
            log += f"   File size: {results['tflite_size_mb']:.2f} MB\n"
            log += f"   Path: {results['tflite_path']}\n"
        elif export_tflite:
            tflite_error = results.get("tflite_error", "Unknown conversion error")
            log += f"\n❌ TFLite export failed: {tflite_error}\n"
            log += f"   Hint: Ensure 'onnx2tf' is correctly installed and the model architecture is supported.\n"

        if results.get("fixed_path"):
            log += f"\n✅ Copied to ESPHome fixed path:\n"
            log += f"   {results['fixed_path']}\n"
        elif esphome_compatible:
            log += f"\n❌ Failed to copy to ESPHome fixed path: {results.get('fixed_path_error', 'Unknown error')}\n"

        log += f"\n" + "=" * 60 + "\n"
        log += f"✅ Export complete!\n"

        status = f"✅ Export Successful\n"
        status += f"Model: {results['architecture']}\n"
        status += f"File: {output_filename} ({results['file_size_mb']:.2f} MB)"

        if quantize_fp16 and "fp16_path" in results:
            status += f"\nFP16: {results['fp16_size_mb']:.2f} MB ({results['fp16_reduction']:.1f}% smaller)"

        if quantize_int8 and "int8_path" in results:
            status += f"\nINT8: {results['int8_size_mb']:.2f} MB ({results['int8_reduction']:.1f}% smaller)"

        if results.get("tflite_success", False):
            status += f"\nTFLite: {results['tflite_size_mb']:.2f} MB"
            if results.get("fixed_path"):
                status += " (ESPHome Ready)"

        # Size warnings
        if results.get("size_warning", False):
            status = "⚠️ Export Successful (Size Warning)\n" + status[18:]
            status += "\n\n⚠️ WARNING: Model exceeds target Flash/RAM limits! Check logs for details."

        logger.info("Export complete")

        return status, log

    except Exception as e:
        error_msg = f"❌ Export failed: {str(e)}"
        logger.error(error_msg)
        logger.exception(e)

        log = f"[{time.strftime('%H:%M:%S')}] ERROR\n"
        log += f"{str(e)}\n"

        return error_msg, log


def validate_exported_model(output_filename: str) -> Tuple[Dict, pd.DataFrame]:
    """
    Validate exported ONNX model

    Args:
        output_filename: ONNX model filename

    Returns:
        Tuple of (model_info_dict, performance_dataframe)
    """
    if not export_state.last_export_path or not export_state.last_checkpoint:
        return {"status": "❌ No model exported yet. Export a model first."}, None

    try:
        logger.info(f"Validating ONNX model: {export_state.last_export_path}")

        # Load PyTorch model for comparison
        checkpoint = torch.load(export_state.last_checkpoint, map_location="cuda")
        config_data = checkpoint["config"]

        # Convert config dict to WakewordConfig object if needed
        from src.config.defaults import WakewordConfig

        if isinstance(config_data, dict):
            config = WakewordConfig.from_dict(config_data)
            logger.info("Converted config dict to WakewordConfig object")
        else:
            config = config_data

        from src.models.architectures import create_model

        # Calculate input size for model
        input_samples = int(config.data.sample_rate * config.data.audio_duration)
        time_steps = input_samples // config.data.hop_length + 1

        feature_dim = (
            config.data.n_mels
            if config.data.feature_type == "mel_spectrogram" or config.data.feature_type == "mel"
            else config.data.n_mfcc
        )

        if config.model.architecture == "cd_dnn":
            input_size = feature_dim * time_steps
        else:
            input_size = feature_dim

        pytorch_model = create_model(
            architecture=config.model.architecture,
            num_classes=config.model.num_classes,
            pretrained=False,
            dropout=config.model.dropout,
            input_size=input_size,
            input_channels=1,
            # RNN (LSTM/GRU) params
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
            bidirectional=config.model.bidirectional,
            # TCN / TinyConv params
            tcn_num_channels=getattr(config.model, "tcn_num_channels", None),
            tcn_kernel_size=getattr(config.model, "tcn_kernel_size", 3),
            tcn_dropout=getattr(config.model, "tcn_dropout", config.model.dropout),
            # CD-DNN params
            cddnn_hidden_layers=getattr(config.model, "cddnn_hidden_layers", None),
            cddnn_context_frames=getattr(config.model, "cddnn_context_frames", 50),
            cddnn_dropout=getattr(config.model, "cddnn_dropout", config.model.dropout),
        )

        # Load weights
        state_dict = checkpoint["model_state_dict"]

        # Handle QAT checkpoints loaded into FP32 models
        # Filter out quantization keys that are not in the model
        model_keys = set(pytorch_model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        unexpected_keys = checkpoint_keys - model_keys

        if unexpected_keys:
            # Check if these are quantization keys
            quant_keys = [
                k for k in unexpected_keys if "fake_quant" in k or "activation_post_process" in k or "observer" in k
            ]

            if quant_keys:
                logger.warning(f"Filtering out {len(quant_keys)} quantization keys from state_dict for FP32 loading")
                # Filter the state dict
                state_dict = {k: v for k, v in state_dict.items() if k in model_keys}

        pytorch_model.load_state_dict(state_dict, strict=True)
        pytorch_model.to("cuda")
        pytorch_model.eval()

        # Create sample input
        sample_rate = config.data.sample_rate
        duration = config.data.audio_duration
        n_mels = config.data.n_mels
        hop_length = config.data.hop_length

        n_samples = int(sample_rate * duration)
        n_frames = n_samples // hop_length + 1

        sample_input = torch.randn(1, 1, n_mels, n_frames).to("cuda")

        # Validate ONNX model
        validation_results = validate_onnx_model(
            onnx_path=export_state.last_export_path,
            pytorch_model=pytorch_model,
            sample_input=sample_input,
            device="cuda",
        )

        export_state.validation_results = validation_results

        # Build model info
        model_info = {
            "Status": "✅ Valid" if validation_results["valid"] else "❌ Invalid",
            "File Size": f"{validation_results.get('file_size_mb', 0):.2f} MB",
            "Graph Nodes": validation_results.get("graph", 0),
            "Input Shape": str(validation_results.get("inputs", [])),
            "Output Shape": str(validation_results.get("outputs", [])),
        }

        if validation_results.get("inference_success", False):
            model_info["Inference"] = "✅ Success"

        if validation_results.get("numerically_equivalent", False):
            model_info["Numerical Match"] = f"✅ Max diff: {validation_results['max_difference']:.6f}"
        elif "max_difference" in validation_results:
            model_info["Numerical Match"] = f"⚠️ Max diff: {validation_results['max_difference']:.6f}"

        # Benchmark if validation successful
        if validation_results.get("valid", False):
            logger.info("Running performance benchmark...")

            benchmark_results = benchmark_onnx_model(
                onnx_path=export_state.last_export_path,
                pytorch_model=pytorch_model,
                sample_input=sample_input,
                num_runs=100,
                device="cuda",
            )

            export_state.benchmark_results = benchmark_results

            # Build performance comparison table
            perf_data = []

            perf_data.append(
                {
                    "Framework": "PyTorch (FP32)",
                    "Inference Time (ms)": f"{benchmark_results['pytorch_time_ms']:.2f}",
                    "Speedup": "1.00x",
                }
            )

            perf_data.append(
                {
                    "Framework": "ONNX",
                    "Inference Time (ms)": f"{benchmark_results['onnx_time_ms']:.2f}",
                    "Speedup": f"{benchmark_results['speedup']:.2f}x",
                }
            )

            perf_df = pd.DataFrame(perf_data)

            logger.info("Validation and benchmarking complete")

            return model_info, perf_df

        else:
            return model_info, None

    except Exception as e:
        error_msg = f"❌ Validation failed: {str(e)}"
        logger.error(error_msg)
        logger.exception(e)

        return {"status": error_msg}, None


def download_onnx_model() -> Tuple[Optional[str], str]:
    """
    Prepare ONNX model for download

    Returns:
        Tuple of (file_path, status_message)
    """
    if not export_state.last_export_path:
        return None, "❌ No model exported yet"

    if not export_state.last_export_path.exists():
        return None, f"❌ Model file not found: {export_state.last_export_path}"

    logger.info(f"Preparing download: {export_state.last_export_path}")

    return (
        str(export_state.last_export_path),
        f"✅ Ready to download: {export_state.last_export_path.name}",
    )


def create_export_panel() -> gr.Blocks:
    """
    Create Panel 5: ONNX Export

    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks() as panel:
        gr.Markdown("# 📦 ONNX Export")
        gr.Markdown("Convert trained PyTorch models to ONNX format for deployment with quantization options.")

        gr.Markdown("### Select Model Checkpoint")

        with gr.Row():
            checkpoint_selector = gr.Dropdown(
                choices=get_available_checkpoints(),
                label="Model Checkpoint",
                info="Select a trained model to export",
                value=(
                    get_available_checkpoints()[0]
                    if get_available_checkpoints()[0] != "No checkpoints available"
                    else None
                ),
            )
            refresh_checkpoints_btn = gr.Button("🔄 Refresh", scale=0)

        gr.Markdown("---")

        gr.Markdown("### Export Configuration")

        with gr.Row():
            with gr.Column():
                gr.Markdown("**Basic Settings**")

                output_filename = gr.Textbox(
                    label="Output Filename",
                    value="wakeword_model.onnx",
                    placeholder="model.onnx",
                    info="Exported model filename",
                )

                opset_version = gr.Dropdown(
                    choices=[11, 12, 13, 14, 15, 16],
                    value=14,
                    label="ONNX Opset Version",
                    info="Version 14 recommended for best compatibility",
                )

                dynamic_batch = gr.Checkbox(
                    label="Dynamic Batch Size",
                    value=True,
                    info="Allow variable batch size during inference (recommended)",
                )

            with gr.Column():
                gr.Markdown("**Quantization Options**")

                quantize_fp16 = gr.Checkbox(
                    label="FP16 Quantization (Float16)",
                    value=False,
                    info="Half precision: ~50% smaller, minimal accuracy loss",
                )

                quantize_int8 = gr.Checkbox(
                    label="INT8 Quantization (8-bit Integer)",
                    value=False,
                    info="8-bit: ~75% smaller, slight accuracy loss",
                )

                export_tflite = gr.Checkbox(
                    label="Export to TFLite (via onnx2tf)",
                    value=False,
                    info="Convert ONNX to TFLite for embedded devices",
                )

                esphome_compatible = gr.Checkbox(
                    label="ESPHome Atom Echo Compatibility",
                    value=False,
                    info="Export to fixed path for Atom Echo firmware",
                )

                gr.Markdown("**Note**: Quantization reduces model size and improves inference speed")

        with gr.Row():
            export_btn = gr.Button("🚀 Export to ONNX", variant="primary", scale=2)
            validate_btn = gr.Button("✅ Validate ONNX", variant="secondary", scale=1)

        gr.Markdown("---")

        gr.Markdown("### Export Status")

        with gr.Row():
            export_status = gr.Textbox(
                label="Status",
                value="Ready to export. Select a checkpoint and configure settings above.",
                lines=5,
                interactive=False,
            )

        with gr.Row():
            export_log = gr.Textbox(
                label="Export Log",
                lines=12,
                value="Configure export settings and click 'Export to ONNX' to begin...\n",
                interactive=False,
                autoscroll=True,
            )

        gr.Markdown("---")

        gr.Markdown("### Validation & Performance")

        with gr.Row():
            with gr.Column():
                gr.Markdown("**Model Information**")
                model_info = gr.JSON(
                    label="ONNX Model Details",
                    value={"status": "Export a model to see details"},
                )

            with gr.Column():
                gr.Markdown("**Performance Comparison**")
                performance_comparison = gr.Dataframe(
                    headers=["Framework", "Inference Time (ms)", "Speedup"],
                    label="PyTorch vs ONNX Benchmark",
                    interactive=False,
                )

        with gr.Row():
            download_file = gr.File(label="Download ONNX Model", visible=False)
            download_btn = gr.Button("⬇️ Download ONNX Model", variant="primary")

        # Event handlers
        def refresh_checkpoints_handler() -> Any:
            checkpoints = get_available_checkpoints()
            return gr.update(
                choices=checkpoints,
                value=checkpoints[0] if checkpoints[0] != "No checkpoints available" else None,
            )

        refresh_checkpoints_btn.click(fn=refresh_checkpoints_handler, outputs=[checkpoint_selector])

        export_btn.click(
            fn=export_to_onnx,
            inputs=[
                checkpoint_selector,
                output_filename,
                opset_version,
                dynamic_batch,
                quantize_fp16,
                quantize_int8,
                export_tflite,
                esphome_compatible,
            ],
            outputs=[export_status, export_log],
        )

        validate_btn.click(
            fn=validate_exported_model,
            inputs=[output_filename],
            outputs=[model_info, performance_comparison],
        )

        def download_handler() -> Tuple[Optional[str], str, Dict[str, Any]]:
            file_path, status = download_onnx_model()
            if file_path:
                return file_path, status, gr.update(visible=True, value=file_path)
            else:
                return None, status, gr.update(visible=False)

        download_btn.click(fn=download_handler, outputs=[download_file, export_status, download_file])

    return panel


if __name__ == "__main__":
    # Test the panel
    demo = create_export_panel()
    demo.launch()
