# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""APIs to create and manage tensors in a graph."""

from .io_spec import (
    FusedInput,
    FusedOutput,
    Input,
    IOSpec,
    IOUnknown,
    MutableInput,
    Output,
)
from .managed_tensor_slice import (
    DynamicTensor,
    InputTensor,
    InputVariadicTensors,
    ManagedTensorSlice,
    OutputTensor,
    OutputVariadicTensors,
    StaticTensorSpec,
    VariadicTensors,
    _input_fusion_hook_impl,
    _output_fusion_hook_impl,
    foreach,
    simd_load_from_managed_tensor_slice,
    simd_store_into_managed_tensor_slice,
    view_copy_impl,
)
from .tensor import Tensor
from .tensor_shape import TensorShape
from .tensor_spec import RuntimeTensorSpec, TensorSpec
