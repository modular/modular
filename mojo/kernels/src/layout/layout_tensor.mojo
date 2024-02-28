# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import DTypePointer

from .int_tuple import flatten, int
from .layout import *


@register_passable
struct StaticLayout:
    var shape: StaticIntTuple[2]
    var stride: StaticIntTuple[2]

    @always_inline
    fn __init__(shape: StaticIntTuple[2], stride: StaticIntTuple[2]) -> Self:
        return Self {shape: shape, stride: stride}

    @always_inline
    fn __init__(layout: Layout) -> Self:
        if len(layout.shape) != 2 or len(layout.stride) != 2:
            trap("Unsupported Layout dimensions.")

        return Self {
            shape: StaticIntTuple[2](
                int(layout.shape[0]), int(layout.shape[1])
            ),
            stride: StaticIntTuple[2](
                int(layout.stride[0]), int(layout.stride[1])
            ),
        }

    @always_inline
    fn __copyinit__(existing: Self) -> Self:
        return Self {shape: existing.shape, stride: existing.stride}

    @always_inline
    fn to_layout(self) -> Layout:
        return Layout(
            IntTuple(self.shape[0], self.shape[1]),
            IntTuple(self.stride[0], self.stride[1]),
        )


struct LayoutTensor[dtype: DType, M: Int, N: Int](CollectionElement):
    var ptr: DTypePointer[dtype]
    var is_view: Bool
    var layout: StaticLayout

    @always_inline
    fn __init__(inout self, layout: Layout, ptr: DTypePointer[dtype]):
        self.ptr = ptr
        self.is_view = True
        self.layout = StaticLayout(layout)
        if self.dim(0) != M or self.dim(1) != N:
            trap("Layout inconsistent with dimensions.")

    @always_inline
    fn __init__(inout self):
        self.ptr = DTypePointer[dtype].alloc(M * N)
        self.is_view = False
        self.layout = StaticLayout((M, N), (N, 1))
        if self.dim(0) != M or self.dim(1) != N:
            trap("Layout inconsistent with dimensions.")

    @always_inline
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
    fn _offset(self, m: Int, n: Int) -> Int:
        var stride = self.layout.stride
        return stride[0] * m + stride[1] * n

    @always_inline
    fn __getitem__(self, m: Int, n: Int) -> Scalar[dtype]:
        return self.ptr.simd_load[1](self._offset(m, n))

    @always_inline
    fn __setitem__(self, m: Int, n: Int, val: Scalar[dtype]):
        self.ptr.simd_store[1](self._offset(m, n), val)

    @always_inline
    fn load[width: Int](self, m: Int, n: Int) -> SIMD[dtype, width]:
        return self.ptr.simd_load[width](self._offset(m, n))

    @always_inline
    fn store[width: Int](self, m: Int, n: Int, val: SIMD[dtype, width]):
        return self.ptr.simd_store[width](self._offset(m, n), val)

    @always_inline
    fn dim(self, idx: Int) -> Int:
        var shape = self.layout.shape
        return shape[idx]

    fn view[
        M1: Int, N1: Int  # View's dimensions
    ](self, m: Int, n: Int) -> LayoutTensor[dtype, M1, N1]:
        var tiler = MakeLayoutList(Layout(M1, 1), Layout(N1, 1))
        var tiled_layout = zipped_divide(self.layout.to_layout(), tiler)
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
                self.layout.to_layout(),
                Layout(IntTuple(N, M), IntTuple(M, 1)),
            ),
            self.ptr,
        )

    @always_inline
    fn copy_from(self, other: Self):
        for m in range(M):
            for n in range(N):
                self[m, n] = other[m, n]

    @always_inline
    fn linspace(self):
        for m in range(M):
            for n in range(N):
                self[m, n] = m * M + n

    @always_inline
    fn fill(self, val: Scalar[dtype]):
        for m in range(M):
            for n in range(N):
                self[m, n] = val

    fn print(self):
        for m in range(M):
            for n in range(N):
                print_no_newline(self[m, n], "  ")
            print("")
