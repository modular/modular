# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import DTypePointer

from .int_tuple import flatten, int
from .layout import *


@register_passable
struct StaticLayout[size: Int = 2]:
    var shape: StaticIntTuple[size]
    var stride: StaticIntTuple[size]

    @always_inline
    fn __init__(
        shape: StaticIntTuple[size], stride: StaticIntTuple[size]
    ) -> Self:
        return Self {shape: shape, stride: stride}

    @always_inline
    fn __init__(layout: Layout) -> Self:
        if len(layout.shape) != size or len(layout.stride) != size:
            trap("Unsupported Layout dimensions.")

        var shape = StaticIntTuple[size]()
        var stride = StaticIntTuple[size]()

        @unroll
        for i in range(size):
            shape[i] = int(layout.shape[i])
            stride[i] = int(layout.stride[i])

        return Self {shape: shape, stride: stride}

    @always_inline
    fn __copyinit__(existing: Self) -> Self:
        return Self {shape: existing.shape, stride: existing.stride}

    @always_inline
    fn to_layout(self) -> Layout:
        var shape = IntTuple()
        var stride = IntTuple()

        for i in range(size):
            shape.append(self.shape[i])
            stride.append(self.stride[i])

        return Layout(shape, stride)


fn NewLayoutTensor[
    M: Int,
    N: Int,
    dtype: DType,
    layout: Layout = Layout(IntTuple(M, N), IntTuple(N, 1)),
]() -> LayoutTensor[layout, dtype]:
    return LayoutTensor[layout, dtype]()


struct LayoutTensor[layout: StaticLayout, dtype: DType](CollectionElement):
    var ptr: DTypePointer[dtype]
    var is_view: Bool

    @always_inline
    fn __init__(inout self):
        self.ptr = DTypePointer[dtype].alloc(self.dim(0) * self.dim(1))
        self.is_view = False

    @always_inline
    fn __init__(inout self, ptr: DTypePointer[dtype]):
        self.ptr = ptr
        self.is_view = True

    @always_inline
    fn __del__(owned self):
        if not self.is_view:
            self.ptr.free()

    @always_inline
    fn __copyinit__(inout self: Self, existing: Self):
        self.ptr = existing.ptr
        self.is_view = True

    @always_inline
    fn __moveinit__(inout self: Self, owned existing: Self):
        self.ptr = existing.ptr
        self.is_view = True

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
    @staticmethod
    fn dim(idx: Int) -> Int:
        return layout.shape[idx]

    @staticmethod
    fn _compute_layout[layout: StaticLayout, M: Int, N: Int]() -> Layout:
        alias tiler = MakeLayoutList(Layout(M, 1), Layout(N, 1))
        return zipped_divide(layout.to_layout(), tiler)

    fn view[
        M1: Int,
        N1: Int,
        tiled_layout: Layout = Self._compute_layout[layout, M1, N1](),
    ](self, m: Int, n: Int) -> LayoutTensor[tiled_layout[0], dtype]:
        # TODO: Figure out how expensive this actually is and optimize it
        # var offset = inner_product(IntTuple(m, n), tiled_layout[1].stride)
        alias inner_tile = StaticLayout(tiled_layout[1])
        var offset = m * inner_tile.stride[0] + n * inner_tile.stride[1]
        return LayoutTensor[tiled_layout[0], dtype](self.ptr.offset(offset))

    fn transpose[
        M: Int = layout.shape[0],
        N: Int = layout.shape[1],
        transposed_layout: Layout = composition(
            layout.to_layout(),
            Layout(IntTuple(N, M), IntTuple(M, 1)),
        ),
    ](self) -> LayoutTensor[transposed_layout, dtype]:
        return LayoutTensor[transposed_layout, dtype](self.ptr)

    @always_inline
    fn copy_from[
        other_layout: StaticLayout
    ](self, other: LayoutTensor[other_layout, dtype]):
        alias M = layout.shape[0]
        alias N = layout.shape[1]
        for m in range(M):
            for n in range(N):
                self[m, n] = other[m, n]

    fn linspace(self):
        alias M = layout.shape[0]
        alias N = layout.shape[1]
        for m in range(M):
            for n in range(N):
                self[m, n] = m * N + n

    fn fill(self, val: Scalar[dtype]):
        alias M = layout.shape[0]
        alias N = layout.shape[1]
        for m in range(M):
            for n in range(N):
                self[m, n] = val

    fn print(self):
        alias M = layout.shape[0]
        alias N = layout.shape[1]
        for m in range(M):
            for n in range(N):
                print_no_newline(self[m, n], "  ")
            print("")
