# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import DTypePointer

from .int_tuple import flatten, int
from .layout import *


struct LayoutTensor[dtype: DType, M: Int, N: Int](CollectionElement):
    var ptr: DTypePointer[dtype]
    var is_view: Bool
    var layout: Layout

    @always_inline
    fn __init__(inout self, layout: Layout, ptr: DTypePointer[dtype]):
        self.ptr = ptr
        self.is_view = True
        self.layout = layout
        if self.dim(0) != M or self.dim(1) != N:
            trap("Layout inconsistent with dimensions.")

    @always_inline
    fn __init__(inout self):
        self.ptr = DTypePointer[dtype].alloc(M * N)
        self.is_view = False
        self.layout = Layout(IntTuple(M, N), IntTuple(N, 1))
        if self.dim(0) != M or self.dim(1) != N:
            trap("Layout inconsistent with dimensions.")

    fn __del__(owned self):
        if not self.is_view:
            self.ptr.free()

    @always_inline
    fn __copyinit__(inout self: Self, existing: Self):
        self.ptr = existing.ptr
        self.is_view = True
        self.layout = existing.layout

    @always_inline
    fn __moveinit__(inout self: Self, owned existing: Self):
        self.ptr = existing.ptr
        self.is_view = True
        self.layout = existing.layout ^

    @always_inline
    fn __getitem__(self, idx: IntTuple) -> Scalar[dtype]:
        return self.ptr.simd_load[1](self.layout(idx))

    @always_inline
    fn __setitem__(self, idx: IntTuple, val: Scalar[dtype]):
        self.ptr.simd_store[1](self.layout(idx), val)

    @always_inline
    fn __getitem__(self, m: Int, n: Int) -> Scalar[dtype]:
        return self.ptr.simd_load[1](self.layout(IntTuple(m, n)))

    @always_inline
    fn __setitem__(self, m: Int, n: Int, val: Scalar[dtype]):
        self.ptr.simd_store[1](self.layout(IntTuple(m, n)), val)

    fn load[width: Int](self, m: Int, n: Int) -> SIMD[dtype, width]:
        return self.ptr.simd_load[width](self.layout(IntTuple(m, n)))

    fn store[width: Int](self, m: Int, n: Int, val: SIMD[dtype, width]):
        return self.ptr.simd_store[width](self.layout(IntTuple(m, n)), val)

    @always_inline
    fn dim(self, idx: Int) -> Int:
        return int(flatten(self.layout.shape)[idx])

    fn view[
        M1: Int, N1: Int  # View's dimensions
    ](self, m: Int, n: Int) -> LayoutTensor[dtype, M1, N1]:
        alias tiler = LayoutList(Layout(M1, 1), Layout(N1, 1))
        var tiled_layout = zipped_divide(self.layout, tiler)
        var coords = IntTuple(m, n)
        if len(coords) > 0:
            var offset = inner_product(coords, tiled_layout[1].stride)
            var res_tensor = LayoutTensor[dtype, M1, N1](
                tiled_layout[0], self.ptr.offset(offset)
            )
            return res_tensor
        return LayoutTensor[dtype, M1, N1](tiled_layout, self.ptr)

    fn transpose(self) -> LayoutTensor[dtype, N, M]:
        return LayoutTensor[dtype, N, M](
            composition(
                self.layout,
                Layout(IntTuple(N, M), IntTuple(M, 1)),
            ),
            self.ptr,
        )

    fn copy_from(self, other: Self):
        for m in range(M):
            for n in range(N):
                self[IntTuple(m, n)] = other[IntTuple(m, n)]

    fn linspace(self):
        for m in range(M):
            for n in range(N):
                self[IntTuple(m, n)] = m * M + n

    fn fill(self, val: Scalar[dtype]):
        for m in range(M):
            for n in range(N):
                self[IntTuple(m, n)] = val

    fn print(self):
        for m in range(M):
            for n in range(N):
                print_no_newline(self[IntTuple(m, n)], "  ")
            print("")
