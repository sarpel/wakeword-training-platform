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
    if config.backend == "qnnpack":
        torch.backends.quantized.engine = "qnnpack"
    elif config.backend == "fbgemm":
        torch.backends.quantized.engine = "fbgemm"
    else:
        logger.warning(f"Unknown QAT backend: {config.backend}. Defaulting to fbgemm.")
        torch.backends.quantized.engine = "fbgemm"

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
