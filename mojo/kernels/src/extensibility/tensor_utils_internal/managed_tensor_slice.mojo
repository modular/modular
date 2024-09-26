# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .indexing import _dot_prod, _slice_to_tuple, _row_major_strides
from .tensor_like import TensorLike

from tensor_internal import TensorSpec, StaticTensorSpec

from collections import InlineArray, OptionalReg
from utils import StaticIntTuple
from memory import UnsafePointer
from math import ceil
from sys import simdwidthof
from sys.intrinsics import strided_load, strided_store
import algorithm
from runtime.asyncrt import MojoCallContextPtr

from compiler_internal.directives import __mogg_intrinsic_attr, specsof
from compiler_internal.directives import StaticTensorSpec as CompilerTensorSpec


@value
@register_passable
struct ManagedTensorSlice[
    type: DType,
    rank: Int,
](TensorLike):
    """ManagedTensorSlice is like TensorSlice but it does not effect the life
    of the underlying allocated pointer. Unlike TensorSlice, when the object
    lifetime ends it does not effect the lifetime of the underlying pointer.
    Conversly, if a ManagedTensorSlice is created, it will not extend the life
    of the underlying pointer.

    Therefore, the user must take care to keep the ptr alive until
    ManagedTensorSlice's last use. This class is useful for writing kernels
    where memory is managed by an external runtime like in MAX's inference
    stack.
    """

    var _ptr: UnsafePointer[Scalar[type]]
    var _spec: StaticTensorSpec[type, rank]
    var _start_offset: Int
    var _strides: StaticIntTuple[rank]

    fn __init__(
        inout self,
        ptr: UnsafePointer[Scalar[type]],
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
        ptr: UnsafePointer[Scalar[type]],
        shape: StaticIntTuple[rank],
    ):
        self._ptr = ptr
        self._spec = StaticTensorSpec[type, rank](shape)
        self._strides = _row_major_strides(self._spec)
        self._start_offset = 0

    fn __init__(
        inout self,
        ptr: UnsafePointer[Scalar[type]],
        shape: StaticIntTuple[rank],
        strides: StaticIntTuple[rank],
    ):
        self = Self(
            ptr,
            StaticTensorSpec[type, rank](shape),
            0,
            strides,
        )

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

    fn dim_size(self, index: Int) -> Int:
        return self._spec.shape[index]

    fn unsafe_ptr[__type: DType = type](self) -> UnsafePointer[Scalar[__type]]:
        return rebind[UnsafePointer[Scalar[__type]]](self._ptr)

    @always_inline
    fn load[
        width: Int,
        # Necessary to make it simpler on the call site.
        _rank: Int,
    ](self, index: StaticIntTuple[_rank]) -> SIMD[type, width]:
        constrained[_rank == rank]()
        var ridx = rebind[StaticIntTuple[rank]](index)
        return self._simd_load_internal[width](ridx)

    @always_inline
    fn _fused_load[
        width: Int,
        # Necessary to make it simpler on the call site.
        _rank: Int,
    ](self, index: StaticIntTuple[_rank]) -> SIMD[type, width]:
        # Nop function to preserve symbols from DCE.
        self._input_fusion_hook[CompilerTensorSpec[type, rank]()]()

        constrained[_rank == rank]()
        var ridx = rebind[StaticIntTuple[rank]](index)

        alias in_lambda = specsof[type, rank]("self").in_lambda

        @parameter
        if in_lambda:
            alias in_fn = in_lambda.value()
            return in_fn[width](ridx)
        else:
            return self._simd_load_internal[width](ridx)

    @always_inline
    fn _simd_load_internal[
        width: Int,
    ](self, index: StaticIntTuple[rank]) -> SIMD[type, width]:
        var flat_index = _dot_prod(index, self._strides)
        var stride = self._strides[rank - 1]

        if stride == 0:
            return self._ptr.load(flat_index)
        elif stride == 1:

            @parameter
            if type is DType.bool:
                var v = self._ptr.bitcast[DType.uint8]().load[width=width](
                    flat_index
                )
                return v.cast[type]()
            else:
                return self._ptr.load[width=width](flat_index)
        else:

            @parameter
            if type is DType.bool:
                var v = strided_load[width](
                    self._ptr.bitcast[DType.uint8]().offset(flat_index),
                    stride,
                )
                return v.cast[type]()
            else:
                return strided_load[width](self._ptr.offset(flat_index), stride)

    @always_inline
    fn store[
        width: Int,
        # Necessary to make it simpler on the call site.
        _rank: Int,
    ](self, index: StaticIntTuple[_rank], val: SIMD[type, width]):
        constrained[_rank == rank]()
        var ridx = rebind[StaticIntTuple[rank]](index)
        self._simd_store_internal[width](ridx, val)

    @always_inline
    fn _fused_store[
        width: Int,
        # Necessary to make it simpler on the call site.
        _rank: Int,
    ](self, index: StaticIntTuple[_rank], val: SIMD[type, width]):
        # Nop function to preserve symbols from DCE.
        self._output_fusion_hook[CompilerTensorSpec[type, rank]()]()

        constrained[_rank == rank]()
        var ridx = rebind[StaticIntTuple[rank]](index)

        alias out_lambda = specsof[type, rank]("self").out_lambda

        @parameter
        if out_lambda:
            alias out_fn = out_lambda.value()
            out_fn[width](ridx, val)
        else:
            self._simd_store_internal[width](ridx, val)

    @always_inline
    fn _simd_store_internal[
        width: Int,
    ](self, index: StaticIntTuple[rank], val: SIMD[type, width]):
        var flat_index = _dot_prod(index, self._strides)

        var stride = self._strides[rank - 1]
        if stride == 0:
            self._ptr.store[width=1](flat_index)
        elif stride == 1:

            @parameter
            if type is DType.bool:
                var v = val.cast[DType.uint8]()
                self._ptr.bitcast[DType.uint8]().store[width = val.size](
                    flat_index, v
                )
            else:
                self._ptr.store(flat_index, val)
        else:

            @parameter
            if type is DType.bool:
                var v = val.cast[DType.uint8]()
                strided_store(
                    v,
                    self._ptr.bitcast[DType.uint8]().offset(flat_index),
                    stride,
                )
            else:
                return strided_store(val, self._ptr.offset(flat_index), stride)

    # Helper function used in SliceMOGGDPSFunc to generate a new TensorSpec with the lambda.
    @no_inline
    fn _extract_new_spec[T: CompilerTensorSpec[type, rank]](self):
        pass

    # Helper function used in SliceMOGGDPSFunc to generate the body of the input lambda
    @__mogg_intrinsic_attr("mogg.dps_input_fusion_hook")
    @no_inline
    fn _input_fusion_hook[spec: CompilerTensorSpec[type, rank]](self):
        @always_inline
        @parameter
        fn _input_lambda[_w: Int](i: StaticIntTuple[rank]) -> SIMD[type, _w]:
            return rebind[SIMD[type, _w]](self._simd_load_internal[_w](i))

        self._extract_new_spec[
            CompilerTensorSpec[type, rank](
                spec.shape, spec.strides, _input_lambda, None
            )
        ]()

    # Helper function used in SliceMOGGDPSFunc to generate the body of the output lambda
    @__mogg_intrinsic_attr("mogg.dps_output_fusion_hook")
    @no_inline
    fn _output_fusion_hook[spec: CompilerTensorSpec[type, rank]](self):
        @always_inline
        @parameter
        fn _output_lambda[_w: Int](i: StaticIntTuple[rank], v: SIMD[type, _w]):
            self._simd_store_internal(i, rebind[SIMD[type, _w]](v))

        self._extract_new_spec[
            CompilerTensorSpec[type, rank](
                spec.shape,
                spec.strides,
                None,
                _output_lambda,
            )
        ]()


# This version of the function supports CPU only. For GPU, use the one with the
# MojoCallContextPtr.
@__mogg_intrinsic_attr("mogg.for_each")
@no_inline
fn foreach[
    type: DType,
    rank: Int, //,
    func: fn[width: Int] (StaticIntTuple[rank]) capturing -> SIMD[type, width],
    synchronous: Bool = False,
    target: StringLiteral = "cpu",
](tensor: ManagedTensorSlice[type, rank]):
    alias simd_width = simdwidthof[tensor.type]()

    @parameter
    fn elementwise_fn_wrapper[
        width: Int, rank: Int
    ](index: StaticIntTuple[rank]) capturing:
        var val = func[width](rebind[StaticIntTuple[tensor.rank]](index))
        tensor._fused_store(index, val)

    algorithm.functional.elementwise[
        elementwise_fn_wrapper,
        simd_width,
        use_blocking_impl=synchronous,
        target=target,
    ](tensor.get_static_spec().shape)


@__mogg_intrinsic_attr("mogg.for_each")
@no_inline
fn foreach[
    type: DType,
    rank: Int, //,
    func: fn[width: Int] (StaticIntTuple[rank]) capturing -> SIMD[type, width],
    synchronous: Bool = False,
    target: StringLiteral = "cpu",
](tensor: ManagedTensorSlice[type, rank], ctx: MojoCallContextPtr):
    alias simd_width = simdwidthof[tensor.type]()

    @parameter
    fn elementwise_fn_wrapper[
        width: Int, rank: Int
    ](index: StaticIntTuple[rank]) capturing:
        var val = func[width](rebind[StaticIntTuple[tensor.rank]](index))
        tensor.store(index, val)

    algorithm.functional.elementwise[
        elementwise_fn_wrapper,
        simd_width,
        use_blocking_impl=synchronous,
        target=target,
    ](tensor.get_static_spec().shape, ctx)
