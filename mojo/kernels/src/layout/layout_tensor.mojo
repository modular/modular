# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import DTypePointer

from .int_tuple import flatten, int
from .layout import *


@register_passable
struct LayoutTensor[layout: Layout, dtype: DType](CollectionElement):
    var ptr: DTypePointer[dtype]

    @always_inline
    fn __init__(inout self, ptr: DTypePointer[dtype]):
        self.ptr = ptr

    @always_inline
    fn __copyinit__(inout self: Self, existing: Self):
        self.ptr = existing.ptr

    @always_inline
    fn _offset(self, m: Int, n: Int) -> Int:
        return Self.stride[0]() * m + Self.stride[1]() * n

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

    @staticmethod
    fn _toStatic[t: IntTuple]() -> StaticIntTuple[len(t)]:
        var st = StaticIntTuple[len(t)]()
        for i in range(len(t)):
            st[i] = int(t[i])
        return st

    @always_inline
    @staticmethod
    fn shape[idx: Int]() -> Int:
        alias shape = Self._toStatic[layout.shape]()
        return shape[idx]

    @always_inline
    @staticmethod
    fn stride[idx: Int]() -> Int:
        alias stride = Self._toStatic[layout.stride]()
        return stride[idx]

    @always_inline
    @staticmethod
    fn dim[idx: Int]() -> Int:
        return Self.shape[idx]()

    @staticmethod
    fn _compute_tile_layout[layout: Layout, M: Int, N: Int]() -> Layout:
        alias tiler = MakeLayoutList(Layout(M, 1), Layout(N, 1))
        return zipped_divide(layout, tiler)

    fn view[
        M1: Int,
        N1: Int,
        tiled_layout: Layout = Self._compute_tile_layout[layout, M1, N1](),
    ](self, m: Int, n: Int) -> LayoutTensor[tiled_layout[0], dtype]:
        # var offset = inner_product(IntTuple(m, n), tiled_layout[1].stride)
        alias tiled_layout_stride = Self._toStatic[tiled_layout[1].stride]()
        var offset = m * tiled_layout_stride[0] + n * tiled_layout_stride[1]
        return LayoutTensor[tiled_layout[0], dtype](self.ptr.offset(offset))

    fn transpose[
        M: Int = Self.dim[0](),
        N: Int = Self.dim[1](),
        transposed_layout: Layout = composition(
            layout,
            Layout(IntTuple(N, M), IntTuple(M, 1)),
        ),
    ](self) -> LayoutTensor[transposed_layout, dtype]:
        return LayoutTensor[transposed_layout, dtype](self.ptr)

    @always_inline
    fn copy_from[
        other_layout: Layout
    ](self, other: LayoutTensor[other_layout, dtype]):
        for m in range(Self.dim[0]()):
            for n in range(Self.dim[1]()):
                self[m, n] = other[m, n]

    fn linspace(self):
        for m in range(Self.dim[0]()):
            for n in range(Self.dim[1]()):
                self[m, n] = m * Self.dim[1]() + n

    fn fill(self, val: Scalar[dtype]):
        for m in range(Self.dim[0]()):
            for n in range(Self.dim[1]()):
                self[m, n] = val

    fn print(self):
        for m in range(Self.dim[0]()):
            for n in range(Self.dim[1]()):
                print_no_newline(self[m, n], "  ")
            print("")
