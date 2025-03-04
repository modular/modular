# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""APIs to create and manage tensors in a graph."""

# Don't put public documentation strings in this file. The docs build does not
# recognize these re-exported definitions, so it won't pick up an docs here.
# The docs are generated from the original implementations.

from tensor import (
    ManagedTensorSlice,
    StaticTensorSpec,
    InputTensor,
    OutputTensor,
    RuntimeTensorSpec,
    Tensor,
    TensorShape,
    TensorSpec,
    foreach,
)
