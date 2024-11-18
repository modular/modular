# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from bit import is_power_of_two
from collections import InlineArray, OptionalReg
from math import ceil, fma
from sys import alignof, simdwidthof
from sys.intrinsics import strided_load, strided_store

import algorithm
from compiler_internal.directives import StaticTensorSpec
from compiler_internal.directives import __mogg_intrinsic_attr, specsof
from memory import UnsafePointer
from runtime.asyncrt import MojoCallContextPtr
from tensor_internal import RuntimeTensorSpec, TensorSpec

from buffer import NDBuffer
from utils import IndexList

from .indexing import _dot_prod, _row_major_strides, _slice_to_tuple
from .tensor_like import TensorLike


@value
@register_passable("trivial")
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
    var _spec: RuntimeTensorSpec[type, rank]
    var _start_offset: Int
    var _strides: IndexList[rank]

    fn __init__(
        inout self,
        ptr: UnsafePointer[Scalar[type]],
        slices: InlineArray[Slice, rank],
        slicer_spec: RuntimeTensorSpec[type, rank],
    ):
        @parameter
        @always_inline
        fn start_fn(slice: Slice) -> Int:
            return slice.start.value()

        @parameter
        @always_inline
        fn stop_fn(slice: Slice) -> Int:
            return slice.end.value()

        @parameter
        @always_inline
        fn step_fn(slice: Slice) -> Int:
            return slice.step.or_else(1)

        var start = _slice_to_tuple[start_fn](slices)
        var stop = _slice_to_tuple[stop_fn](slices)
        var step = _slice_to_tuple[step_fn](slices)

        var adjusted_shape = IndexList[rank]()
        for i in range(rank):
            adjusted_shape[i] = int(ceil((stop[i] - start[i]) / step[i]))
        var slice_spec = RuntimeTensorSpec[type](adjusted_shape)

        var slicer_strides = _row_major_strides(slicer_spec)
        var start_offset = _dot_prod(start, slicer_strides)

        var strides = IndexList[rank]()

        @parameter
        for i in range(rank):
            strides[i] = step[i] * slicer_strides[i]

        self = Self(ptr, slice_spec, start_offset, strides)

    fn __init__(
        inout self,
        ptr: UnsafePointer[Scalar[type]],
        shape: IndexList[rank],
    ):
        self._ptr = ptr
        self._spec = RuntimeTensorSpec[type, rank](shape)
        self._strides = _row_major_strides(self._spec)
        self._start_offset = 0

    fn __init__(
        inout self,
        ptr: UnsafePointer[Scalar[type]],
        shape: IndexList[rank],
        strides: IndexList[rank],
    ):
        self = Self(
            ptr,
            RuntimeTensorSpec[type, rank](shape),
            0,
            strides,
        )

    @implicit
    fn __init__(out self, ndbuffer: NDBuffer[type, rank]):
        """Initializes a ManagedTensorSlice from an NDBuffer.

        Note that forwarding of static shape, strides, and lambdas won't work.
        """
        self = Self(ndbuffer.data, ndbuffer.get_shape())

    @always_inline
    fn get_runtime_spec(self) -> RuntimeTensorSpec[type, rank]:
        """Gets the static spec of the slice.

        Returns:
            Static tensor spec of slice.
        """
        return self._spec

    @always_inline
    fn __getitem__(self, indices: IndexList[rank]) -> Scalar[type]:
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
    fn __setitem__(self, indices: IndexList[rank], val: Scalar[type]):
        """Stores the value at the specified indices.

        Args:
          indices: The indices of the value to store.
          val: The value to store.

        """
        var offset = self._start_offset + _dot_prod(indices, self._strides)
        self._ptr[offset] = val

    @always_inline
    fn spec(self) -> TensorSpec:
        return self._spec.get_tensor_spec()

    @always_inline
    fn dim_size(self, index: Int) -> Int:
        return self._spec.shape[index]

    @always_inline
    fn size(self) -> Int:
        """Computes the tensor slice's number of elements.

        Returns:
            The total number of elements in the ManagedTensorSlice.
        """
        var product: Int = 1

        @parameter
        for i in range(rank):
            product *= self.dim_size(i)

        return product

    @always_inline
    fn unsafe_ptr[__type: DType = type](self) -> UnsafePointer[Scalar[__type]]:
        return rebind[UnsafePointer[Scalar[__type]]](self._ptr)

    @always_inline
    fn load[
        width: Int,
        # Necessary to make it simpler on the call site.
        _rank: Int,
    ](self, index: IndexList[_rank]) -> SIMD[type, width]:
        constrained[_rank == rank]()
        var ridx = rebind[IndexList[rank]](index)
        alias alignment = specsof[type, rank]("self").alignment
        return self._simd_load_internal[width, alignment=alignment](ridx)

    @always_inline
    fn _fused_load[
        width: Int,
        # Necessary to make it simpler on the call site.
        _rank: Int,
    ](self, index: IndexList[_rank]) -> SIMD[type, width]:
        # Nop function to preserve symbols from DCE.
        self._input_fusion_hook()

        constrained[_rank == rank]()
        var ridx = rebind[IndexList[rank]](index)

        alias in_lambda = specsof[type, rank]("self").in_lambda
        alias alignment = specsof[type, rank]("self").alignment

        @parameter
        if in_lambda:
            alias in_fn = in_lambda.value()
            return in_fn[width](ridx)
        else:
            return self._simd_load_internal[width, alignment](ridx)

    @always_inline
    fn _compute_offset(self, index: IndexList[rank]) -> Int:
        # TODO(GRA-1017): Add address space cases.
        @parameter
        if rank == 0:
            return 0

        var offset = 0

        @parameter
        for i in range(rank):
            offset = fma(index[i], self._strides[i], offset)

        return offset

    @always_inline
    fn _simd_load_internal[
        width: Int,
        alignment: Int = 1,
    ](self, index: IndexList[rank]) -> SIMD[type, width]:
        var flat_index = self._compute_offset(index)
        var stride = self._strides[rank - 1]

        # Load alignment cannot exceed the data type's alignment.
        alias max_alignment = gcd_pow2[alignment, alignof[type]()]()

        if stride == 0:
            return self._ptr.load(flat_index)
        elif stride == 1:

            @parameter
            if type is DType.bool:
                var v = self._ptr.bitcast[Scalar[DType.uint8]]().load[
                    width=width
                ](flat_index)
                return v.cast[type]()
            else:
                return self._ptr.load[width=width, alignment=max_alignment](
                    flat_index
                )
        else:

            @parameter
            if type is DType.bool:
                var v = strided_load[width](
                    self._ptr.bitcast[Scalar[DType.uint8]]().offset(flat_index),
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
    ](self, index: IndexList[_rank], val: SIMD[type, width]):
        constrained[_rank == rank]()
        var ridx = rebind[IndexList[rank]](index)

        alias alignment = specsof[type, rank]("self").alignment
        self._simd_store_internal[width, alignment=alignment](ridx, val)

    @always_inline
    fn _fused_store[
        width: Int,
        # Necessary to make it simpler on the call site.
        _rank: Int,
    ](self, index: IndexList[_rank], val: SIMD[type, width]):
        # Nop function to preserve symbols from DCE.
        self._output_fusion_hook()

        constrained[_rank == rank]()
        var ridx = rebind[IndexList[rank]](index)

        alias out_lambda = specsof[type, rank]("self").out_lambda
        alias alignment = specsof[type, rank]("self").alignment

        @parameter
        if out_lambda:
            alias out_fn = out_lambda.value()
            out_fn[width](ridx, val)
        else:
            self._simd_store_internal[width, alignment=alignment](ridx, val)

    @always_inline
    fn _simd_store_internal[
        width: Int,
        alignment: Int = 1,
    ](self, index: IndexList[rank], val: SIMD[type, width]):
        var flat_index = self._compute_offset(index)

        # Store alignment cannot exceed the data type's alignment.
        alias max_alignment = gcd_pow2[alignment, alignof[type]()]()

        var stride = self._strides[rank - 1]
        if stride == 0:
            self._ptr[] = flat_index
        elif stride == 1:

            @parameter
            if type is DType.bool:
                var v = val.cast[DType.uint8]()
                self._ptr.bitcast[Scalar[DType.uint8]]().store(flat_index, v)
            else:
                self._ptr.store[alignment=max_alignment](flat_index, val)
        else:

            @parameter
            if type is DType.bool:
                var v = val.cast[DType.uint8]()
                strided_store(
                    v,
                    self._ptr.bitcast[Scalar[DType.uint8]]().offset(flat_index),
                    stride,
                )
            else:
                return strided_store(val, self._ptr.offset(flat_index), stride)

    # Helper functions used in SliceMOGGDPSFunc to ensure the lambda isn't DCE
    @no_inline
    fn _extract_lambda[T: StaticTensorSpec[type, rank].in_lambda_t](self):
        pass

    @no_inline
    fn _extract_lambda[T: StaticTensorSpec[type, rank].out_lambda_t](self):
        pass

    # Helper function used in SliceMOGGDPSFunc to generate the body of the input lambda
    @__mogg_intrinsic_attr("mogg.dps_input_fusion_hook")
    @no_inline
    fn _input_fusion_hook(self):
        @always_inline
        @parameter
        fn _input_lambda[_w: Int](i: IndexList[rank]) -> SIMD[type, _w]:
            return rebind[SIMD[type, _w]](self._simd_load_internal[_w](i))

        self._extract_lambda[_input_lambda]()

    # Helper function used in SliceMOGGDPSFunc to generate the body of the output lambda
    @__mogg_intrinsic_attr("mogg.dps_output_fusion_hook")
    @no_inline
    fn _output_fusion_hook(self):
        @always_inline
        @parameter
        fn _output_lambda[_w: Int](i: IndexList[rank], v: SIMD[type, _w]):
            self._simd_store_internal(i, rebind[SIMD[type, _w]](v))

        self._extract_lambda[_output_lambda]()


@parameter
@always_inline
fn gcd_pow2[a: Int, b: Int]() -> Int:
    # alignments should always be powers of 2
    constrained[
        is_power_of_two(a) and is_power_of_two(b),
        "a and b must be powers of 2",
    ]()
    return min(a, b)


# This version of the function supports CPU only. For GPU, use the one with the
# MojoCallContextPtr.
@__mogg_intrinsic_attr("mogg.for_each")
@no_inline
fn foreach[
    type: DType,
    rank: Int, //,
    func: fn[width: Int] (IndexList[rank]) capturing -> SIMD[type, width],
    synchronous: Bool = False,
    target: StringLiteral = "cpu",
](tensor: ManagedTensorSlice[type, rank]):
    alias simd_width = simdwidthof[tensor.type]()

    @parameter
    @always_inline
    fn elementwise_fn_wrapper[
        width: Int, rank: Int
    ](index: IndexList[rank]) capturing:
        var val = func[width](rebind[IndexList[tensor.rank]](index))
        tensor._fused_store(index, val)

    algorithm.functional.elementwise[
        elementwise_fn_wrapper,
        simd_width,
        use_blocking_impl=synchronous,
        target=target,
    ](tensor.get_runtime_spec().shape)


@__mogg_intrinsic_attr("mogg.for_each")
@no_inline
fn foreach[
    type: DType,
    rank: Int, //,
    func: fn[width: Int] (IndexList[rank]) capturing -> SIMD[type, width],
    synchronous: Bool = False,
    target: StringLiteral = "cpu",
](tensor: ManagedTensorSlice[type, rank], ctx: MojoCallContextPtr):
    alias simd_width = simdwidthof[tensor.type]()

    @parameter
    @always_inline
    fn elementwise_fn_wrapper[
        width: Int, rank: Int
    ](index: IndexList[rank]) capturing:
        var val = func[width](rebind[IndexList[tensor.rank]](index))
        tensor._fused_store(index, val)

    algorithm.functional.elementwise[
        elementwise_fn_wrapper,
        simd_width,
        use_blocking_impl=synchronous,
        target=target,
    ](tensor.get_runtime_spec().shape, ctx)
