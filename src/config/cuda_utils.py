"""
CUDA Detection and Validation Utilities
No CPU fallback for tensor operations - GPU is mandatory
"""
import torch
import sys
from typing import Dict, Tuple, Any


class CUDAValidator:
    """Validates CUDA availability and provides GPU information"""

    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.cuda_available else 0

    def validate(self) -> Tuple[bool, str]:
        """
        Validate CUDA setup

        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        if not self.cuda_available:
            return False, (
                "❌ CUDA is not available. GPU is MANDATORY for this platform.\n"
                "Please ensure:\n"
                "  1. NVIDIA GPU is installed\n"
                "  2. CUDA Toolkit is installed (11.8 or 12.x)\n"
                "  3. PyTorch is installed with CUDA support\n"
                "  4. GPU drivers are up to date\n"
            )

        if self.device_count == 0:
            return False, (
                "❌ No CUDA devices detected.\n"
                "CUDA is available but no GPU devices found.\n"
                "Please check your GPU installation."
            )

        return True, f"✅ CUDA validated successfully. {self.device_count} GPU(s) available."

    def get_device_info(self) -> Dict[str, Any]:
        """
        Get detailed GPU information

        Returns:
            Dict containing GPU information
        """
        if not self.cuda_available:
            return {
                "cuda_available": False,
                "device_count": 0,
                "devices": [],
                "error": "CUDA not available"
            }

        devices = []
        for i in range(self.device_count):
            props = torch.cuda.get_device_properties(i)
            devices.append({
                "id": i,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
                "multi_processor_count": props.multi_processor_count,
            })

        return {
            "cuda_available": True,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": self.device_count,
            "devices": devices,
            "current_device": torch.cuda.current_device(),
        }

    def get_memory_info(self, device_id: int = 0) -> Dict[str, float]:
        """
        Get GPU memory information

        Args:
            device_id: GPU device ID

        Returns:
            Dict with memory statistics in GB
        """
        if not self.cuda_available:
            return {"error": "CUDA not available"}

        torch.cuda.set_device(device_id)
        total = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
        reserved = torch.cuda.memory_reserved(device_id) / (1024**3)
        free = total - allocated

        return {
            "device_id": device_id,
            "total_gb": round(total, 2),
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "free_gb": round(free, 2),
            "utilization_percent": round((allocated / total) * 100, 2),
        }

    def estimate_batch_size(self, model_size_mb: float = 50,
                           sample_size_mb: float = 0.5,
                           device_id: int = 0) -> int:
        """
        Estimate safe batch size based on available GPU memory

        Args:
            model_size_mb: Estimated model size in MB
            sample_size_mb: Estimated size per sample in MB
            device_id: GPU device ID

        Returns:
            Recommended batch size
        """
        if not self.cuda_available:
            return 0

        mem_info = self.get_memory_info(device_id)
        available_gb = mem_info["free_gb"]

        # Reserve 20% for safety and gradients
        usable_gb = available_gb * 0.8
        usable_mb = usable_gb * 1024

        # Account for model size
        available_for_data = usable_mb - model_size_mb

        if available_for_data <= 0:
            return 1

        # Calculate batch size (multiply by 2 for gradients)
        batch_size = int(available_for_data / (sample_size_mb * 2))

        # Clamp between reasonable values
        return max(1, min(batch_size, 256))

    def clear_cache(self):
        """Clear CUDA cache"""
        if self.cuda_available:
            torch.cuda.empty_cache()

    def get_device(self, device_id: int = 0) -> torch.device:
        """
        Get torch device (GPU only, no CPU fallback)

        Args:
            device_id: GPU device ID

        Returns:
            torch.device for CUDA

        Raises:
            RuntimeError: If CUDA is not available
        """
        if not self.cuda_available:
            raise RuntimeError(
                "GPU is MANDATORY for this platform. CUDA is not available.\n"
                "Please install CUDA and PyTorch with GPU support."
            )

        if device_id >= self.device_count:
            raise ValueError(
                f"Invalid device_id {device_id}. "
                f"Available devices: 0-{self.device_count-1}"
            )

        return torch.device(f"cuda:{device_id}")


def get_cuda_validator() -> CUDAValidator:
    """Get singleton CUDA validator instance"""
    return CUDAValidator()


def enforce_cuda():
    """
    Enforce CUDA availability at startup
    Exit if CUDA is not available
    """
    validator = CUDAValidator()
    is_valid, message = validator.validate()

    if not is_valid:
        print(message)
        print("\n" + "="*60)
        print("CUDA VALIDATION FAILED - EXITING")
        print("="*60)
        sys.exit(1)

    print(message)

    # Print GPU info
    info = validator.get_device_info()
    print(f"\nCUDA Version: {info['cuda_version']}")
    print(f"cuDNN Version: {info['cudnn_version']}")
    print(f"\nAvailable GPUs:")
    for device in info['devices']:
        print(f"  [{device['id']}] {device['name']}")
        print(f"      Compute Capability: {device['compute_capability']}")
        print(f"      Memory: {device['total_memory_gb']} GB")
        print(f"      Multiprocessors: {device['multi_processor_count']}")

    return validator


if __name__ == "__main__":
    # Test CUDA validation
    enforce_cuda()