# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import fma
from sys.info import simdwidthof
from sys.intrinsics import strided_load

from algorithm.functional import elementwise, vectorize
from buffer import NDBuffer
from buffer.list import DimList
from memory.unsafe import Pointer, bitcast
from register import *
from .tensor_helpers import UnsafeRefCounter, InnerStride

from collections import OptionalReg as Optional


@always_inline
@export
fn empty_tensor[
    type: DType, rank: Int
](shape: StaticIntTuple[rank]) -> Tensor[type, rank]:
    var ptr = DTypePointer[type].alloc(shape.flattened_length())
    var ref_cnt = Pointer[Scalar[DType.index]].alloc(1)
    ref_cnt[0] = 0
    return Tensor[type, rank](ptr, shape, ref_cnt)


struct Tensor[type: DType, static_rank: Int]:
    var data: DTypePointer[type]
    var shape: StaticIntTuple[static_rank]
    var strides: StaticIntTuple[static_rank]
    var storage_ref_count: UnsafeRefCounter[DType.index]

    # Empty strides...
    @always_inline
    fn __init__(
        inout self,
        ptr: DTypePointer[type],
        shape: StaticIntTuple[static_rank],
        ref_count_ptr: Pointer[Scalar[DType.index]],
    ):
        self.data = ptr
        self.shape = shape
        self.strides = StaticIntTuple[static_rank]()

        var stride = 1

        # Walk backwards to compute the fully contiguous strides.
        @unroll
        for i in range(static_rank - 1, -1, -1):
            self.strides[i] = stride
            stride *= self.shape[i]

        # Increment the refcount if we own the memory otherwise create an
        # empty counter.
        self.storage_ref_count = UnsafeRefCounter[DType.index](ref_count_ptr)
        _ = self.storage_ref_count.increment()

    @always_inline
    fn __init__(
        inout self,
        ptr: DTypePointer[type],
        shape: StaticIntTuple[static_rank],
        strides: StaticIntTuple[static_rank],
        ref_count_ptr: Pointer[Scalar[DType.index]],
    ):
        self.data = ptr
        self.shape = shape
        self.strides = strides

        # Increment the refcount if we own the memory otherwise create an
        # empty counter.
        self.storage_ref_count = UnsafeRefCounter[DType.index](ref_count_ptr)
        _ = self.storage_ref_count.increment()

    @no_inline
    fn __copyinit__(inout self, existing: Self):
        self.storage_ref_count = existing.storage_ref_count
        _ = self.storage_ref_count.increment()

        self.data = existing.data
        self.shape = existing.shape
        self.strides = existing.strides

    @always_inline
    fn refcount(self) -> Pointer[SIMD[DType.index, 1]]:
        return self.storage_ref_count._underlying_value

    fn __del__(owned self):
        var res = self.storage_ref_count.decrement()
        if res == 1:
            # Clean up the managed refcount itself
            self.storage_ref_count.deallocate()

            # TODO: Invoke parameterized deconstructor.
            self.data.free()

    @always_inline
    fn nelems(self) -> Int:
        return self.shape.flattened_length()

    @always_inline
    fn rank(self) -> Int:
        return static_rank

    @always_inline
    fn _compute_flat_index(
        self,
        index: StaticIntTuple[static_rank],
    ) -> Int:
        var flat_index: Int = 0

        @always_inline
        @parameter
        fn body[idx: Int]():
            flat_index = fma(index[idx], self.strides[idx], flat_index)

        unroll[body, static_rank]()
        return flat_index

    @always_inline
    fn store[
        width: Int
    ](inout self, index: StaticIntTuple[static_rank], value: SIMD[type, width]):
        self._simd_store_internal(index, value)

    @always_inline
    fn store[width: Int](inout self, index: Int, value: SIMD[type, width]):
        constrained[
            self.static_rank == 1,
            (
                "Single int access to kernels only allowed on tensors"
                " statically known to be 1D"
            ),
        ]()
        var as_nd = self.get_nd_indices()
        as_nd[0] = index
        self.store(as_nd, value)

    @always_inline
    fn _simd_store_internal[
        width: Int
    ](inout self, index: StaticIntTuple[static_rank], val: SIMD[type, width]):
        var flat_index = self._compute_flat_index(index)
        self.data.store[width = val.size](flat_index, val)

    @always_inline
    fn get_nd_indices(self) -> StaticIntTuple[static_rank]:
        return StaticIntTuple[static_rank](0)

    @always_inline
    fn simd_load[simd_width: Int](self, index: Int) -> SIMD[type, simd_width]:
        constrained[
            static_rank == 1,
            (
                "Single int access to kernels only allowed on tensors"
                " statically known to be 1D"
            ),
        ]()
        var as_nd = self.get_nd_indices()
        as_nd[0] = index
        return self.simd_load[simd_width](as_nd)

    @always_inline
    fn simd_load[
        simd_width: Int,
    ](self, index: StaticIntTuple[static_rank]) -> SIMD[type, simd_width]:
        return self._simd_load_internal[simd_width](index)

    @always_inline
    fn _simd_load_internal[
        simd_width: Int
    ](self, index: StaticIntTuple[static_rank]) -> SIMD[type, simd_width]:
        var flat_index = self._compute_flat_index(index)
        var stride = self.strides[self.rank() - 1]

        @parameter
        @always_inline
        fn _load[
            stride_type: InnerStride
        ](stride: Int) -> SIMD[type, simd_width]:
            @parameter
            if stride_type == InnerStride.Broadcast:
                return self.data.load(flat_index)
            elif stride_type == InnerStride.Contiguous:

                @parameter
                if type == DType.bool:
                    var v = self.data.bitcast[DType.uint8]().load[
                        width=simd_width
                    ](flat_index)
                    return v.cast[type]()
                else:
                    return self.data.load[width=simd_width](flat_index)
            else:

                @parameter
                if type == DType.bool:
                    var v = strided_load[DType.uint8, simd_width](
                        self.data.bitcast[DType.uint8]().offset(flat_index),
                        stride,
                    )
                    return v.cast[type]()
                else:
                    return strided_load[type, simd_width](
                        self.data.offset(flat_index), stride
                    )

        if self.strides[self.rank() - 1] == 0:
            return _load[InnerStride.Broadcast](stride)
        elif stride > 1:
            return _load[InnerStride.Strided](stride)
        return _load[InnerStride.Contiguous](stride)

    @no_inline
    fn for_each[
        func: fn[_width: Int] (StaticIntTuple[static_rank]) capturing -> SIMD[
            type, _width
        ],
    ](inout self):
        alias simd_width = simdwidthof[Self.type]()

        @always_inline
        @parameter
        fn elementwise_fn_wrapper[
            width: Int, rank: Int
        ](coords_static: StaticIntTuple[rank]) capturing:
            var coords = rebind[StaticIntTuple[static_rank]](coords_static)
            var val = func[width](coords)
            self.store(coords, val)

        elementwise[elementwise_fn_wrapper, simd_width, static_rank](self.shape)
