# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Represents a sliced view of a tensor. The slice has origin same as that of the
tensor from which it is created.

For example, you can create a TensorSlice and use it like this:

```mojo
from max.driver import Tensor
from max.tensor import TensorShape

def main():
    tensor = Tensor[DType.float32, rank=3](TensorShape(1, 2, 3))
    slice_one = tensor[:]
```

"""
from collections import InlineArray
from math import ceil
from sys import is_nvidia_gpu
from sys.intrinsics import strided_load, strided_store

from max._tensor_utils import DynamicTensor
from max.tensor import RuntimeTensorSpec, TensorSpec

from .tensor import Tensor


@value
@register_passable
struct TensorSlice[
    is_mutable: Bool, //,
    type: DType,
    rank: Int,
    origin: Origin[is_mutable],
]:
    """Sliced view of a tensor. This is safe to use even after the last use of
    tensor from which it is created. For creating a slice use the __getitem__
    method defined in tensor.
    """

    var _ref: Pointer[Tensor[type, rank], origin]
    var _unsafe_slice: DynamicTensor[type, rank].Type

    @doc_private
    fn __init__(
        mut self,
        ref [origin]tensor: Tensor[type, rank],
        slices: InlineArray[Slice, rank],
    ):
        self = Self(
            Pointer.address_of(tensor),
            DynamicTensor[type, rank].Type(tensor._ptr, slices, tensor._spec),
        )

    fn runtime_spec(self) -> RuntimeTensorSpec[type, rank]:
        """Gets the static spec of the slice.

        Returns:
            Static tensor spec of slice.
        """
        return self._unsafe_slice._spec

    fn spec(self) -> TensorSpec:
        """Gets the spec of the slice.

        Returns:
            Spec of slice as TensorSpec.
        """
        return TensorSpec(self._unsafe_slice.spec())

    @always_inline
    fn __getitem__(self, *indices: Int) -> Scalar[type]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        debug_assert(
            len(indices) == rank, "mismatch between requested index and rank"
        )

        @always_inline
        @parameter
        fn _indexible() -> Bool:
            return is_nvidia_gpu() or "cpu" in String(self._ref[]._device)

        debug_assert[_indexible](
            "Cannot index into non-CPU Tensor from host",
        )
        return self._unsafe_slice[indices]

    @always_inline
    fn __setitem__(self, *indices: Int, val: Scalar[type]):
        """Sets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.
          val: The value to store at the specified indices.
        """

        debug_assert(
            len(indices) == rank, "mismatch between requested index and rank"
        )

        @always_inline
        @parameter
        fn _is_cpu() -> Bool:
            return is_nvidia_gpu() or "cpu" in String(self._ref[]._device)

        debug_assert[_is_cpu](
            "Cannot index into non-CPU Tensor from host",
        )
        self._unsafe_slice[indices] = val
