# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from .tensor import (
    Tensor,
    TensorLike,
    StaticTensorSpec,
    _dot_prod,
    _slice_to_tuple,
    _row_major_strides,
)
from utils import InlineArray
from math import ceil
from max.tensor import TensorSpec
from sys.intrinsics import strided_load, strided_store
from sys.info import triple_is_nvidia_cuda


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

    fn unsafe_ptr[__type: DType = type](self) -> DTypePointer[__type]:
        return rebind[DTypePointer[__type]](self._unsafe_slice._ptr)

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
        debug_assert(
            "CPU" in str(self._ref[]._device) or triple_is_nvidia_cuda(),
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
        debug_assert(
            "CPU" in str(self._ref[]._device) or triple_is_nvidia_cuda(),
            "Cannot index into non-CPU Tensor from host",
        )
        self._unsafe_slice[indices] = val


@value
@register_passable
struct UnsafeTensorSlice[
    type: DType,
    rank: Int,
](TensorLike):
    """UnsafeTensorSlice is like TensorSlice but it contains no reference to
    the original Tensor. Therefore, the user must take care to keep the ptr alive
    until UnsafeTensorSlice's last use.
    """

    var _ptr: DTypePointer[type]
    var _spec: StaticTensorSpec[type, rank]
    var _start_offset: Int
    var _strides: StaticIntTuple[rank]

    fn __init__(
        inout self,
        ptr: DTypePointer[type],
        slices: InlineArray[Slice, rank],
        slicer_spec: StaticTensorSpec[type, rank],
    ):
        @parameter
        fn start_fn(slice: Slice) -> Int:
            return slice.start.value()

        @parameter
        fn stop_fn(slice: Slice) -> Int:
            return slice.end.value()

        @parameter
        fn step_fn(slice: Slice) -> Int:
            return slice.step

        var start = _slice_to_tuple[start_fn](slices)
        var stop = _slice_to_tuple[stop_fn](slices)
        var step = _slice_to_tuple[step_fn](slices)

        var adjusted_shape = StaticIntTuple[rank]()
        for i in range(rank):
            adjusted_shape[i] = int(ceil((stop[i] - start[i]) / step[i]))
        var slice_spec = StaticTensorSpec[type](adjusted_shape)

        var slicer_strides = _row_major_strides(slicer_spec)
        var start_offset = _dot_prod(start, slicer_strides)

        var strides = StaticIntTuple[rank]()

        @parameter
        for i in range(rank):
            strides[i] = step[i] * slicer_strides[i]

        self = Self(ptr, slice_spec, start_offset, strides)

    fn __init__(
        inout self,
        ptr: DTypePointer[type],
        shape: StaticIntTuple[rank],
    ):
        self._ptr = ptr
        self._spec = StaticTensorSpec[type, rank](shape)
        self._strides = _row_major_strides(self._spec)
        self._start_offset = 0

    fn get_static_spec(self) -> StaticTensorSpec[type, rank]:
        """Gets the static spec of the slice.

        Returns:
            Static tensor spec of slice.
        """
        return self._spec

    @always_inline
    fn __getitem__(self, indices: StaticIntTuple[rank]) -> Scalar[type]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        var offset = self._start_offset + _dot_prod(indices, self._strides)
        return self._ptr[offset]

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
        return self[indices]

    @always_inline
    fn __setitem__(self, *indices: Int, val: Scalar[type]):
        """Stores the value at the specified indices.

        Args:
          indices: The indices of the value to store.
          val: The value to store.

        """
        debug_assert(
            len(indices) == rank, "mismatch between requested index and rank"
        )
        self[indices] = val

    @always_inline
    fn __setitem__(self, indices: StaticIntTuple[rank], val: Scalar[type]):
        """Stores the value at the specified indices.

        Args:
          indices: The indices of the value to store.
          val: The value to store.

        """
        var offset = self._start_offset + _dot_prod(indices, self._strides)
        self._ptr[offset] = val

    fn spec(self) -> TensorSpec:
        return self._spec.get_tensor_spec()

    fn unsafe_ptr[__type: DType = type](self) -> DTypePointer[__type]:
        return rebind[DTypePointer[__type]](self._ptr)

    @always_inline
    fn load[
        width: Int,
        # Necessary to make it simpler on the call site.
        _rank: Int,
    ](self, index: StaticIntTuple[_rank]) -> SIMD[type, width]:
        constrained[_rank == rank]()
        var flat_index = _dot_prod(
            rebind[StaticIntTuple[rank]](index), self._strides
        )
        var stride = self._strides[rank - 1]

        if stride == 0:
            return Scalar.load(self._ptr, flat_index)
        elif stride == 1:

            @parameter
            if type is DType.bool:
                var v = SIMD[size=width].load(
                    self._ptr.bitcast[DType.uint8](), flat_index
                )
                return v.cast[type]()
            else:
                return SIMD[size=width].load(self._ptr, flat_index)
        else:

            @parameter
            if type is DType.bool:
                var v = strided_load[DType.uint8, width](
                    self._ptr.bitcast[DType.uint8]().offset(flat_index),
                    stride,
                )
                return v.cast[type]()
            else:
                return strided_load[type, width](
                    self._ptr.offset(flat_index), stride
                )

    @always_inline
    fn store[
        # Necessary to make it simpler on the call site.
        _rank: Int,
        width: Int,
    ](inout self, index: StaticIntTuple[_rank], val: SIMD[type, width]):
        constrained[_rank == rank]()
        var flat_index = _dot_prod(
            rebind[StaticIntTuple[rank]](index), self._strides
        )

        var stride = self._strides[rank - 1]
        if stride == 0:
            Scalar.store(self._ptr, flat_index)
        elif stride == 1:

            @parameter
            if type is DType.bool:
                var v = val.cast[DType.uint8]()
                SIMD[size = val.size].store(
                    self._ptr.bitcast[DType.uint8](), flat_index, v
                )
            else:
                SIMD[size = val.size].store(self._ptr, flat_index, val)
        else:

            @parameter
            if type is DType.bool:
                var v = val.cast[DType.uint8]()
                strided_store[DType.uint8, width](
                    v,
                    self._ptr.bitcast[DType.uint8]().offset(flat_index),
                    stride,
                )
            else:
                return strided_store[type, width](
                    val, self._ptr.offset(flat_index), stride
                )
