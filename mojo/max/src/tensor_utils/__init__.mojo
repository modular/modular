# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Note to make it hard to have circular dependencies, the impl lives elsewhere
from tensor_utils_internal import (
    ManagedTensorSlice,
    TensorLike,
    indexing,
    foreach,
)
