"""
Quantization Aware Training (QAT) Utilities
Handles model preparation and configuration for QAT.
"""

import logging
from typing import Optional

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
        from torch.ao.quantization.fake_quantize import (
            FakeQuantize,
            FusedMovingAvgObsFakeQuantize,
        )
    except ImportError:
        try:
            # Fallback for older PyTorch
            from torch.quantization.fake_quantize import (
                FakeQuantize,
                FusedMovingAvgObsFakeQuantize,
            )
        except ImportError:
            logger.warning("Could not import FusedMovingAvgObsFakeQuantize. Skipping cleanup.")
            return model

    from torch.ao.quantization.observer import (
        MovingAverageMinMaxObserver,
        MovingAveragePerChannelMinMaxObserver,
    )

    # Iterate and replace
    for name, module in model.named_children():
        if isinstance(module, FusedMovingAvgObsFakeQuantize):
            logger.info(f"Replacing fused fake quant: {name}")
            
            # Select appropriate observer and kwargs
            if hasattr(module, "ch_axis"):
                obs_cls = MovingAveragePerChannelMinMaxObserver
                kwargs = {
                    "observer": obs_cls,
                    "quant_min": module.quant_min,
                    "quant_max": module.quant_max,
                    "dtype": module.dtype,
                    "qscheme": module.qscheme,
                    "ch_axis": module.ch_axis,
                }
            else:
                obs_cls = MovingAverageMinMaxObserver
                kwargs = {
                    "observer": obs_cls,
                    "quant_min": module.quant_min,
                    "quant_max": module.quant_max,
                    "dtype": module.dtype,
                    "qscheme": module.qscheme,
                }
            
            new_fq = FakeQuantize(**kwargs)
            
            # Copy state (scale, zero_point)
            new_fq.register_buffer("scale", module.scale)
            new_fq.register_buffer("zero_point", module.zero_point)
            
            # Explicitly set ch_axis on the module if present
            if hasattr(module, "ch_axis"):
                new_fq.ch_axis = module.ch_axis
            
            # Disable observer and enable fake_quant
            new_fq.observer_enabled[0] = 0
            new_fq.fake_quant_enabled[0] = 1
            
            # Replace in parent
            setattr(model, name, new_fq)
        
        else:
            # Recursive call
            cleanup_qat_for_export(module)

    return model