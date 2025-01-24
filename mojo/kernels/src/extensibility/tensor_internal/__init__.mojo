# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the MAX tensor package."""

from .tensor import Tensor
from .tensor_shape import TensorShape
from .tensor_spec import RuntimeTensorSpec, TensorSpec
from .managed_tensor_slice import (
    ManagedTensorSlice,
    foreach,
    view_copy_impl,
    simd_store_into_managed_tensor_slice,
    simd_load_from_managed_tensor_slice,
    _input_fusion_hook_impl,
    _output_fusion_hook_impl,
)
from .tensor_like import TensorLike
