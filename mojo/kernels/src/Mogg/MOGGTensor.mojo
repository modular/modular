# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# A temporary home for the experimental int list type.

from math import fma
from MOGGIntList import IntList
from sys.intrinsics import strided_load
from utils.optional import Optional
from utils._annotations import *


struct Tensor[
    type: DType,
    static_shape: DimList = DimList(),
    static_strides: DimList = DimList(),
]:
    alias static_rank = -1 if len(static_shape) == 0 else len(static_shape)

    var data: DTypePointer[type]
    var shape: IntList[static_shape]
    var strides: IntList[static_strides]
    var dyn_rank: Int

    @always_inline
    fn __init__(
        inout self,
        ptr: DTypePointer[type],
        shape: IntList,
        strides: IntList,
    ):
        self.data = ptr
        self.shape = IntList[static_shape](shape)
        self.strides = IntList[static_strides](strides)
        self.dyn_rank = len(shape)

    @mogg_tensor_move_constructor()
    @always_inline
    fn __moveinit__(inout self, owned existing: Self):
        self.data = existing.data
        self.shape = existing.shape ^
        self.strides = existing.strides ^
        self.dyn_rank = existing.dyn_rank
        existing.data = DTypePointer[type]()

    @always_inline
    fn nelems(self) -> Int:
        return self.shape.nelems()

    @always_inline
    fn rank(self) -> Int:
        @parameter
        if Self.static_rank != -1:
            return Self.static_rank
        return self.dyn_rank

    @always_inline
    fn _compute_flat_index(
        self,
        index: IntList,
    ) -> Int:
        var flat_index: Int = 0

        @parameter
        if Self.static_rank != -1:

            @always_inline
            @parameter
            fn body[idx: Int]():
                flat_index = fma(index[idx], self.strides[idx], flat_index)

            unroll[Self.static_rank, body]()
        else:
            # Dynamic case for dynamic ranks.
            for idx in range(self.dyn_rank):
                flat_index = fma(index[idx], self.strides[idx], flat_index)
        return flat_index

    @always_inline
    fn simd_store[
        width: Int,
    ](self, index: IntList, val: SIMD[type, width]):
        self._simd_store_internal(index, val)

    @always_inline
    fn _simd_store_internal[
        width: Int,
    ](self, index: IntList, val: SIMD[type, width]):
        let flat_index = self._compute_flat_index(index)
        self.data.simd_store[width](flat_index, val)

    @always_inline
    fn simd_load[
        simd_width: Int
    ](self, index: IntList) -> SIMD[type, simd_width]:
        return self._simd_load_internal[simd_width](index)

    @always_inline
    fn _simd_load_internal[
        simd_width: Int
    ](self, index: IntList) -> SIMD[type, simd_width]:
        let flat_index = self._compute_flat_index(index)
        let stride = self.strides[self.rank() - 1]

        if self.strides[self.rank() - 1] == 0:
            return self.data.load(flat_index)
        elif stride > 1:

            @parameter
            if type == DType.bool:
                let v = strided_load[DType.uint8, simd_width](
                    self.data.bitcast[DType.uint8]().offset(flat_index), stride
                )
                return v.cast[type]()
            else:
                return strided_load[type, simd_width](
                    self.data.offset(flat_index), stride
                )

        @parameter
        if type == DType.bool:
            let v = self.data.bitcast[DType.uint8]().simd_load[simd_width](
                flat_index
            )
            return v.cast[type]()
        else:
            return self.data.simd_load[simd_width](flat_index)
