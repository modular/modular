# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""
Implements the `TensorLike` trait, used to share code for objects that
behave like tensors.
"""

from memory import UnsafePointer
from tensor_internal import TensorSpec


trait TensorLike:
    """A trait which exposes functions common to all tensor types."""

    fn spec(self) -> TensorSpec:
        """Gets the `TensorSpec` of this tensor, which provides meta-data
        about the tensor.

        Returns:
            The static `TensorSpec` for this tensor.
        """
        ...

    fn unsafe_ptr[type: DType](self) -> UnsafePointer[Scalar[type]]:
        """Gets the pointer stored in this tensor.

        Danger: You should avoid using this function because the
        returned pointer can modify the invariants of
        this tensor and lead to unexpected behavior.

        Parameters:
            type: The type of the `UnsafePointer` in this tensor.

        Returns:
            The `UnsafePointer` which contains the data for this tensor.
        """
        pass
