# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the MAX tensor package."""

# Note to make it hard to have circular dependencies, the impl lives elsewhere
from tensor_internal.tensor import Tensor
from tensor_internal.tensor_shape import TensorShape
from tensor_internal.tensor_spec import TensorSpec, RuntimeTensorSpec
