# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import DTypePointer

from .int_tuple import flatten, int
from .layout import *


struct LayoutTensor[dtype: DType]:
    var ptr: DTypePointer[dtype]
    var layout: Layout

    @always_inline
    fn __init__(inout self, layout: Layout, ptr: DTypePointer[dtype]):
        self.ptr = ptr
        self.layout = layout

    @always_inline
    fn __init__(inout self, M: Int, N: Int):
        self.ptr = DTypePointer[dtype].alloc(M * N)
        self.layout = Layout(IntTuple(M, N), IntTuple(N, 1))

    @always_inline
    fn __copyinit__(inout self: Self, existing: Self):
        self.ptr = existing.ptr
        self.layout = existing.layout

    @always_inline
    fn __getitem__(self, idx: IntTuple) -> SIMD[dtype, 1]:
        return self.ptr.simd_load[1](self.layout(idx))

    @always_inline
    fn __setitem__(self, idx: IntTuple, val: SIMD[dtype, 1]):
        self.ptr.simd_store[1](self.layout(idx), val)

    @always_inline
    fn dim(self, idx: Int) -> Int:
        return int(flatten(self.layout.shape)[idx])

    fn view(self, tiler: LayoutList, coords: IntTuple) -> LayoutTensor[dtype]:
        var tiled_layout = zipped_divide(self.layout, tiler)
        if len(coords) > 0:
            var offset = inner_product(coords, tiled_layout[1].stride)
            var res_tensor = LayoutTensor[dtype](
                tiled_layout[0], self.ptr.offset(offset)
            )
            return res_tensor
        return LayoutTensor[dtype](tiled_layout, self.ptr)

    fn transpose(self) -> LayoutTensor[dtype]:
        return LayoutTensor(
            composition(
                self.layout,
                Layout(
                    IntTuple(self.dim(1), self.dim(0)), IntTuple(self.dim(0), 1)
                ),
            ),
            self.ptr,
        )

    fn copyTo(self, other: Self):
        if self.dim(0) != other.dim(0) or self.dim(1) != other.dim(1):
            trap(
                String("matrix dimensions don't match: ")
                + self.dim(0)
                + ":"
                + self.dim(1)
                + " != "
                + other.dim(0)
                + ":"
                + other.dim(1)
            )
        for m in range(self.dim(0)):
            for n in range(self.dim(1)):
                other[IntTuple(m, n)] = self[IntTuple(m, n)]
