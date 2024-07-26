# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from .tensor import Tensor
from max.tensor import StaticTensorSpec, TensorSpec
from max.tensor_utils import TensorLike
from tensor_utils.indexing import (
    _dot_prod,
    _slice_to_tuple,
    _row_major_strides,
)
from utils import InlineArray
from math import ceil
from max.tensor import TensorSpec
from sys.intrinsics import strided_load, strided_store


@value
@register_passable
struct TensorSlice[
    is_mutable: Bool, //,
    type: DType,
    rank: Int,
    lifetime: AnyLifetime[is_mutable].type,
](TensorLike):
    alias _ref_type = Reference[Tensor[type, rank], lifetime]
    var _ref: Self._ref_type
    var _unsafe_slice: UnsafeTensorSlice[type, rank]

    fn __init__(
        inout self, tensor: Self._ref_type, slices: InlineArray[Slice, rank]
    ):
        self = Self(
            tensor,
            UnsafeTensorSlice[type, rank](
                tensor[]._ptr, slices, tensor[]._spec
            ),
        )

    fn static_spec(self) -> StaticTensorSpec[type, rank]:
        """Gets the static spec of the slice.

        Returns:
            Static tensor spec of slice.
        """
        return self._unsafe_slice.get_static_spec()

    fn spec(self) -> TensorSpec:
        return self._unsafe_slice._spec.get_tensor_spec()

    fn unsafe_ptr[__type: DType = type](self) -> UnsafePointer[Scalar[__type]]:
        return rebind[UnsafePointer[Scalar[__type]]](self._unsafe_slice._ptr)

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
            return triple_is_nvidia_cuda() or "CPU" in str(self._ref[]._device)

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
            return triple_is_nvidia_cuda() or "CPU" in str(self._ref[]._device)

        debug_assert[_is_cpu](
            "Cannot index into non-CPU Tensor from host",
        )
        self._unsafe_slice[indices] = val
