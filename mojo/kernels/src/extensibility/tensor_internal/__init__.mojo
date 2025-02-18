# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the MAX tensor package."""

from .managed_tensor_slice import (
    DynamicTensor,
    InputTensor,
    ManagedTensorSlice,
    MutableInputTensor,
    OutputTensor,
    VariadicTensors,
    _input_fusion_hook_impl,
    _output_fusion_hook_impl,
    foreach,
    simd_load_from_managed_tensor_slice,
    simd_store_into_managed_tensor_slice,
    view_copy_impl,
)
from .tensor import Tensor
from .tensor_like import TensorLike
from .tensor_shape import TensorShape
from .tensor_spec import RuntimeTensorSpec, TensorSpec
from .io_spec import (
    IOSpec,
    IOUnknown,
    Input,
    Output,
    MutableInput,
)
