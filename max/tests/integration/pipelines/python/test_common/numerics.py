# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Utilities for controlling numerical precision"""

import functools
import threading

import torch

# Add a lock for thread safety
_tf32_lock = threading.RLock()


def pytorch_disable_tf32_dtype(func):  # noqa: ANN001
    """Thread-safe decorator which disables TF32 for PyTorch code.

    PyTorch uses the TensorFloat32 precision by default on modern NVIDIA GPUs.
    MAX uses Float32, and this difference can mask real numerical issues on
    our comparison tests.

    See: https://docs.pytorch.org/docs/stable/notes/cuda.html
    This decorator disables TF32 precision for the decorated function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _tf32_lock:
            # Store original flag values
            original_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
            original_cudnn_tf32 = torch.backends.cudnn.allow_tf32

            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Restore original flag values
                torch.backends.cuda.matmul.allow_tf32 = original_matmul_tf32
                torch.backends.cudnn.allow_tf32 = original_cudnn_tf32

    return wrapper
