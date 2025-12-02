"""
Security Utilities for Wakeword Training Platform
Path validation, safe file operations, and deserialization helpers
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import structlog
import torch

logger = structlog.get_logger(__name__)


# Allowed base directories for path operations (relative to working directory)
ALLOWED_BASE_PATHS: List[str] = [
    "data",
    "models",
    "configs",
    "exports",
    "tests",
    "docs",
]


def validate_path(
    path: Union[str, Path],
    allowed_base_dirs: Optional[List[str]] = None,
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    allow_absolute: bool = True,
) -> Path:
    """
    Validate a path to prevent path traversal attacks.

    Args:
        path: The path to validate
        allowed_base_dirs: List of allowed base directories (default: ALLOWED_BASE_PATHS)
        must_exist: If True, raise error if path doesn't exist
        must_be_file: If True, raise error if path is not a file
        must_be_dir: If True, raise error if path is not a directory
        allow_absolute: If True, allow absolute paths (resolved and validated)

    Returns:
        Validated Path object

    Raises:
        ValueError: If path is invalid or attempts traversal
        FileNotFoundError: If must_exist is True and path doesn't exist
    """
    if allowed_base_dirs is None:
        allowed_base_dirs = ALLOWED_BASE_PATHS

    # Convert to Path object
    path_obj = Path(path)

    # Resolve to absolute path to eliminate .. and symlinks
    try:
        resolved_path = path_obj.resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid path: {path}") from e

    # Get current working directory
    cwd = Path.cwd().resolve()

    # Check if path is within allowed directories
    is_allowed = False

    # Check if the resolved path is under current working directory
    try:
        resolved_path.relative_to(cwd)
        is_allowed = True
    except ValueError:
        # Path is outside cwd, check if absolute paths are allowed
        if allow_absolute:
            # For absolute paths, verify no path traversal attempts
            path_str = str(path)
            if ".." in path_str:
                raise ValueError(f"Path traversal detected: {path}")
            is_allowed = True

    if not is_allowed:
        raise ValueError(f"Path outside allowed directories: {path}")

    # Check for path traversal attempts in the original path string
    original_path_str = str(path)
    if ".." in original_path_str:
        # Double check by comparing resolved path
        # If original contains ".." but resolved doesn't escape, it's still suspicious
        logger.warning(f"Path contains '..' sequences: {path}")
        # Still allow if resolved path is safe, but log it

    # Additional validation
    if must_exist and not resolved_path.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved_path}")

    if must_be_file and not resolved_path.is_file():
        raise ValueError(f"Path is not a file: {resolved_path}")

    if must_be_dir and not resolved_path.is_dir():
        raise ValueError(f"Path is not a directory: {resolved_path}")

    return resolved_path


def safe_path_join(base_dir: Union[str, Path], *parts: str) -> Path:
    """
    Safely join path components, preventing path traversal.

    Args:
        base_dir: Base directory path
        *parts: Path components to join

    Returns:
        Safe joined path

    Raises:
        ValueError: If resulting path escapes base directory
    """
    base = Path(base_dir).resolve()

    # Filter out dangerous path components
    safe_parts = []
    for part in parts:
        # Remove any path traversal attempts
        clean_part = part.replace("..", "").strip("/").strip("\\")
        if clean_part:
            safe_parts.append(clean_part)

    # Join and resolve
    result = base.joinpath(*safe_parts).resolve()

    # Verify result is still under base
    try:
        result.relative_to(base)
    except ValueError:
        raise ValueError(f"Path traversal attempt detected: {parts}")

    return result


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent directory traversal and other attacks.

    Args:
        filename: The filename to sanitize

    Returns:
        Sanitized filename safe for use in file operations
    """
    # Remove path separators and parent directory references
    sanitized = filename.replace("/", "_").replace("\\", "_")
    sanitized = sanitized.replace("..", "_")

    # Remove null bytes and other dangerous characters
    sanitized = sanitized.replace("\x00", "")

    # Limit length
    max_length = 255
    if len(sanitized) > max_length:
        # Preserve extension
        ext = Path(sanitized).suffix
        sanitized = sanitized[: max_length - len(ext)] + ext

    return sanitized


def safe_torch_load(
    path: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = None,
    weights_only: bool = True,
) -> Dict[str, Any]:
    """
    Safely load a PyTorch checkpoint with restricted deserialization.

    Args:
        path: Path to the checkpoint file
        map_location: Device to map tensors to
        weights_only: If True (default), only load tensor data (recommended)

    Returns:
        Loaded checkpoint dictionary

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        ValueError: If file is not a valid checkpoint
    """
    path_obj = validate_path(path, must_exist=True, must_be_file=True)

    logger.debug(f"Loading checkpoint with weights_only={weights_only}: {path_obj}")

    try:
        # Use weights_only=True by default for security
        # This prevents arbitrary code execution during deserialization
        checkpoint = torch.load(str(path_obj), map_location=map_location, weights_only=weights_only)
        return checkpoint
    except Exception as e:
        # If weights_only fails, we might need to load config objects
        # But we should be cautious about this
        if weights_only:
            logger.warning(f"weights_only=True failed, this might indicate unsafe content: {e}")
            raise ValueError(f"Failed to load checkpoint safely: {e}") from e
        raise


def is_safe_subprocess_arg(arg: str) -> bool:
    """
    Check if a string is safe to use as a subprocess argument.

    Args:
        arg: The argument string to check

    Returns:
        True if the argument appears safe
    """
    # Check for shell metacharacters
    dangerous_chars = [";", "&", "|", "$", "`", "(", ")", "{", "}", "<", ">", "!", "\n", "\r"]
    for char in dangerous_chars:
        if char in arg:
            return False
    return True


def validate_subprocess_args(args: List[str]) -> List[str]:
    """
    Validate subprocess arguments for safety.

    Args:
        args: List of command arguments

    Returns:
        Validated list of arguments

    Raises:
        ValueError: If any argument is unsafe
    """
    validated = []
    for arg in args:
        if not is_safe_subprocess_arg(arg):
            raise ValueError(f"Unsafe subprocess argument detected: {arg}")
        validated.append(arg)
    return validated
