# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements CUDA sync operations."""

from ._utils import _check_error, _get_dylib_function

# ===----------------------------------------------------------------------===#
# Synchronize
# ===----------------------------------------------------------------------===#


@always_inline
fn synchronize() raises:
    _check_error(_get_dylib_function[fn () -> Result]("cuCtxSynchronize")())
