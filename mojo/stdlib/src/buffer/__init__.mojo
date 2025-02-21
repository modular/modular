# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the buffer package."""

from .buffer import (  # Explicitly don't import these; partial_simd_store,; prod_dims,
    NDBuffer,
)
from .dimlist import Dim, DimList
