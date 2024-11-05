# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements CUDA sync operations."""

from ._utils_v1 import _check_error
from .cuda_instance_v1 import *

# ===----------------------------------------------------------------------===#
# Synchronize
# ===----------------------------------------------------------------------===#


@always_inline
fn synchronize() raises:
    """Blocks for a Cuda Context's tasks to complete."""
    var cuCtxSynchronize = cuCtxSynchronize.load()
    _check_error(cuCtxSynchronize())
