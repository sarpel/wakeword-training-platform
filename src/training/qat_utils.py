"""
Quantization Aware Training (QAT) Utilities
Handles model preparation and configuration for QAT.
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.quantization

from src.config.defaults import QATConfig

logger = logging.getLogger(__name__)


def prepare_model_for_qat(
    model: nn.Module, config: QATConfig, input_example: Optional[torch.Tensor] = None
) -> nn.Module:
    """
    Prepare a model for Quantization Aware Training.

    Args:
        model: The model to quantize
        config: QAT configuration
        input_example: Example input tensor (optional, for tracing/jit if needed)

    Returns:
        The prepared model
    """
    if not config.enabled:
        return model

    logger.info(f"Preparing model for QAT (Backend: {config.backend})")

    # 1. Set backend
    supported_engines = torch.backends.quantized.supported_engines

    # Determine execution engine
    execution_engine = "none"
    if config.backend in supported_engines:
        execution_engine = config.backend
    elif "fbgemm" in supported_engines:
        execution_engine = "fbgemm"
    elif "x86" in supported_engines:
        execution_engine = "x86"
    elif "onednn" in supported_engines:
        execution_engine = "onednn"

    if execution_engine != config.backend:
        logger.warning(
            f"Requested QAT backend '{config.backend}' is not supported as an execution engine on this machine. "
            f"Using '{execution_engine}' engine for training, but model will be prepared for '{config.backend}'."
        )

    if execution_engine != "none":
        torch.backends.quantized.engine = execution_engine

    # 2. Fuse modules (Optional but recommended for performance/accuracy)
    # Ideally, we would identify the model type and fuse appropriate layers.
    # For standard torchvision models (ResNet, MobileNet), they often have
    # built-in fuse_model() methods if we used the quantized versions,
    # but here we are wrapping standard float models.
    # We will skip manual fusion for now to keep it generic, or add simple
    # pattern matching later if needed.

    # 3. Configure Qconfig
    # Use the default qconfig for the specified backend
    model.qconfig = torch.quantization.get_default_qat_qconfig(config.backend)

    # 4. Prepare
    # prepare_qat inserts observers and fake quantization modules
    # Mypy: Explicitly annotate return type to avoid Any
    prepared_model: nn.Module = torch.quantization.prepare_qat(model, inplace=False)

    logger.info("Model prepared for QAT")
    return prepared_model


def convert_model_to_quantized(model: nn.Module) -> nn.Module:
    """
    Convert a QAT-trained model to a fully quantized model.

    Args:
        model: The QAT-trained model

    Returns:
        The quantized model (INT8)
    """
    model.eval()
    # Mypy: Explicitly annotate return type to avoid Any
    quantized_model: nn.Module = torch.quantization.convert(model, inplace=False)
    return quantized_model


def cleanup_qat_for_export(model: nn.Module) -> nn.Module:
    """
    Replace FusedMovingAvgObsFakeQuantize modules with standard FakeQuantize modules.
    This is necessary for ONNX export, as the 'aten::fused_moving_avg_obs_fake_quant'
    operator is often not supported.

    Args:
        model: The QAT model (in eval mode)

    Returns:
        The cleaned model ready for export
    """
    logger.info("Cleaning up QAT model for ONNX export (replacing Fused ops)...")

    # Try to import the Fused class to check against
    try:
        from torch.ao.quantization.fake_quantize import FakeQuantize, FusedMovingAvgObsFakeQuantize
    except ImportError:
        try:
            # Fallback for older PyTorch
            from torch.quantization.fake_quantize import FakeQuantize, FusedMovingAvgObsFakeQuantize
        except ImportError:
            logger.warning("Could not import FusedMovingAvgObsFakeQuantize. Skipping cleanup.")
            return model

    from torch.ao.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver

    # Iterate and replace
    for name, module in model.named_children():
        if isinstance(module, FusedMovingAvgObsFakeQuantize):
            logger.info(f"Replacing fused fake quant: {name}")

            # Select appropriate observer and kwargs
            # ONNX export often fails with per-channel activation quantization
            # We check if it's weight (ch_axis=0) or activation (ch_axis=-1 or similar)
            is_weight = hasattr(module, "ch_axis") and module.ch_axis == 0

            if is_weight:
                obs_cls: type = MovingAveragePerChannelMinMaxObserver

                # Safeguard: Per-channel observer only supports specific qschemes
                valid_per_ch_schemes = [
                    torch.per_channel_symmetric,
                    torch.per_channel_affine,
                    torch.per_channel_affine_float_qparams,
                ]
                qscheme = module.qscheme
                if qscheme not in valid_per_ch_schemes:
                    qscheme = torch.per_channel_affine

                kwargs = {
                    "observer": obs_cls,
                    "quant_min": module.quant_min,
                    "quant_max": module.quant_max,
                    "dtype": module.dtype,
                    "qscheme": qscheme,
                    "ch_axis": module.ch_axis,
                }
            else:
                # Use per-tensor for activations to avoid ONNX export issues
                obs_cls = MovingAverageMinMaxObserver

                # Per-tensor symmetric or affine
                qscheme = torch.per_tensor_affine
                if module.qscheme == torch.per_tensor_symmetric:
                    qscheme = torch.per_tensor_symmetric

                kwargs = {
                    "observer": obs_cls,
                    "quant_min": module.quant_min,
                    "quant_max": module.quant_max,
                    "dtype": module.dtype,
                    "qscheme": qscheme,
                }

            new_fq = FakeQuantize(**kwargs)

            # Copy state (scale, zero_point)
            if is_weight:
                scale = module.scale
                zp = module.zero_point

                # Check if scale needs expansion (e.g. loaded from FP32 checkpoint)
                # We need to find the channel count. We can peek at the parent or sibling
                # but usually we can infer it from the module being replaced if it's a FQ.
                # However, FQ doesn't know its parent's out_channels directly easily.
                # We can use the fact that if it's a weight FQ, it was replaced in a Conv/Linear.
                # Since we are recursive, we might not have the parent easily.
                # BUT, we can check the size of the weight in the model if we had it.

                # Dynamic repair: If scale is size 1 but we are per-channel,
                # we might need to expand it. We'll try to find a sibling "weight"
                # if this is being called on a module that HAS a weight.
                if scale.numel() == 1 and hasattr(model, "weight"):
                    out_ch = model.weight.shape[0]
                    if out_ch > 1:
                        logger.info(f"Expanding per-channel scale from 1 to {out_ch}")
                        scale = scale.expand(out_ch).clone()
                        zp = zp.expand(out_ch).clone()

                new_fq.register_buffer("scale", scale)
                new_fq.register_buffer("zero_point", zp)
            else:
                # Activation conversion: take first scale value if it was a vector
                scale = module.scale
                if scale.numel() > 1:
                    scale = scale[0:1]
                new_fq.register_buffer("scale", scale)

                zp = module.zero_point
                if zp.numel() > 1:
                    zp = zp[0:1]
                new_fq.register_buffer("zero_point", zp)

            # Disable observer and enable fake_quant
            new_fq.observer_enabled[0] = 0
            new_fq.fake_quant_enabled[0] = 1

            # Replace in parent
            setattr(model, name, new_fq)

        else:
            # Recursive call
            cleanup_qat_for_export(module)

    return model


def calibrate_model(model: nn.Module, data_loader_or_list: Any, device: str = "cpu") -> None:
    """
    Run calibration samples through the model to initialize quantization observers.

    Args:
        model: The prepared QAT model
        data_loader_or_list: DataLoader or list of tensors for calibration
        device: Device to run calibration on
    """
    logger.info("Starting model calibration...")
    model.eval()
    model.to(device)

    with torch.no_grad():
        for i, batch in enumerate(data_loader_or_list):
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch

            inputs = inputs.to(device)
            model(inputs)

            if i >= 100:  # Limit calibration samples
                break

    logger.info("Calibration complete.")


def compare_model_accuracy(
    fp32_model: nn.Module,
    quant_model: nn.Module,
    val_loader_or_list: Any,
    device: str = "cpu",
    audio_processor: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """
    Compare accuracy of FP32 vs Quantized model.

    Args:
        fp32_model: Original float model
        quant_model: Quantized or QAT model
        val_loader_or_list: Validation data
        device: Device to run evaluation on
        audio_processor: Optional GPU audio processor for feature extraction

    Returns:
        Dictionary with accuracy results and drop
    """

    def evaluate(model: nn.Module, data: Any) -> float:
        # Move to CPU for quantized models if requested
        model.eval()

        # Set quantized engine for CPU evaluation
        if device == "cpu":
            model.to("cpu")
            try:
                if "fbgemm" in torch.backends.quantized.supported_engines:
                    torch.backends.quantized.engine = "fbgemm"
                elif "qnnpack" in torch.backends.quantized.supported_engines:
                    torch.backends.quantized.engine = "qnnpack"
            except Exception as e:
                logger.warning(f"Could not set quantized engine: {e}")
        else:
            model.to(device)

        correct = 0
        total = 0
        with torch.no_grad():
            for batch in data:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                else:
                    # Assume it's just inputs (for calibration-like eval)
                    continue

                inputs, targets = inputs.to(device), targets.to(device)

                # Apply audio processor if provided
                if audio_processor is not None:
                    # Ensure processor is on the same device as inputs
                    audio_processor.to(device)
                    audio_processor.eval()
                    inputs = audio_processor(inputs)

                # Ensure inputs are contiguous for quantized ops
                if device == "cpu":
                    inputs = inputs.contiguous()

                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        return correct / total if total > 0 else 0.0

    logger.info("Evaluating FP32 model...")
    # Keep FP32 on its original device for speed if it's CUDA
    fp32_acc = evaluate(fp32_model, val_loader_or_list)

    logger.info("Evaluating Quantized model...")
    quant_acc = evaluate(quant_model, val_loader_or_list)

    drop = fp32_acc - quant_acc

    results = {
        "fp32_acc": fp32_acc,
        "quant_acc": quant_acc,
        "drop": drop,
        "relative_drop": drop / fp32_acc if fp32_acc > 0 else 0.0,
    }

    logger.info(f"Accuracy comparison: FP32={fp32_acc:.4f}, Quant={quant_acc:.4f}, Drop={drop:.4f}")

    return results
