# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Don't put public documentation strings in this file. The docs build does not
# recognize these re-exported definitions, so it won't pick up an docs here.
# The docs are generated from the original implementations.

from tensor_internal import (
    DynamicTensor,
    ManagedTensorSlice,
    TensorLike,
    VariadicTensors,
    _indexing,
    foreach,
)

# Note to make it hard to have circular dependencies, the impl lives elsewhere
from tensor_internal.tensor import Tensor
from tensor_internal.tensor_shape import TensorShape
from tensor_internal.tensor_spec import RuntimeTensorSpec, TensorSpec
