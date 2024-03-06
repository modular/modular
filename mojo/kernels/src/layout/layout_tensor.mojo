# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import DTypePointer

from .int_tuple import flatten, int
from .layout import *
from sys.intrinsics import PrefetchOptions
from algorithm import vectorize
from memory import memcpy


@register_passable
struct LayoutTensor[
    layout: Layout,
    dtype: DType,
    /,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
    owning: Bool = False,
](CollectionElement):
    var ptr: DTypePointer[dtype, address_space]

    @always_inline
    fn __init__(inout self, ptr: DTypePointer[dtype, address_space]):
        self.ptr = ptr

    @always_inline
    fn __copyinit__(inout self: Self, existing: Self):
        self.ptr = existing.ptr

    fn __del__(owned self):
        if owning:
            self.ptr.free()

    @always_inline
    fn _offset(self, m: Int, n: Int) -> Int:
        return Self.stride[0]() * m + Self.stride[1]() * n

    @always_inline
    fn __getitem__(self, *dims: Int) -> Scalar[dtype]:
        # TODO: Static assert ranks are the same!
        alias strides = Self._toStatic[flatten(layout.stride)]()
        return self.ptr.simd_load[1](Self._getOffset(strides, dims))

    @always_inline
    fn __setitem__(self, d0: Int, val: Scalar[dtype]):
        alias strides = Self._toStatic[flatten(layout.stride)]()
        self.ptr.simd_store[1](
            Self._getOffset(strides, VariadicList[Int](d0)), val
        )

    @always_inline
    fn __setitem__(self, d0: Int, d1: Int, val: Scalar[dtype]):
        alias strides = Self._toStatic[flatten(layout.stride)]()
        self.ptr.simd_store[1](
            Self._getOffset(strides, VariadicList[Int](d0, d1)), val
        )

    @always_inline
    fn __setitem__(self, d0: Int, d1: Int, d2: Int, val: Scalar[dtype]):
        alias strides = Self._toStatic[flatten(layout.stride)]()
        self.ptr.simd_store[1](
            Self._getOffset(strides, VariadicList[Int](d0, d1, d2)), val
        )

    @always_inline
    fn __setitem__(
        self, d0: Int, d1: Int, d2: Int, d3: Int, val: Scalar[dtype]
    ):
        alias strides = Self._toStatic[flatten(layout.stride)]()
        self.ptr.simd_store[1](
            Self._getOffset(strides, VariadicList[Int](d0, d1, d2, d3)), val
        )

    @always_inline
    fn load[width: Int](self, m: Int, n: Int) -> SIMD[dtype, width]:
        return self.ptr.simd_load[width](self._offset(m, n))

    @always_inline
    fn load_aligned[width: Int](self, m: Int, n: Int) -> SIMD[dtype, width]:
        alias alignment = alignof[SIMD[dtype, width]]()
        return self.ptr.aligned_simd_load[width, alignment](self._offset(m, n))

    @always_inline
    fn store[width: Int](self, m: Int, n: Int, val: SIMD[dtype, width]):
        return self.ptr.simd_store[width](self._offset(m, n), val)

    @staticmethod
    @always_inline("nodebug")
    fn stack_allocation() -> Self:
        return stack_allocation[
            layout.size(), dtype, address_space=address_space
        ]()

    @staticmethod
    fn _toStatic[t: IntTuple]() -> StaticIntTuple[len(t)]:
        var st = StaticIntTuple[len(t)]()
        for i in range(len(t)):
            st[i] = int(t[i])
        return st

    @staticmethod
    fn _getOffset[
        rank: Int
    ](stride: StaticIntTuple[rank], vals: VariadicList[Int]) -> Int:
        var offset = 0
        for i in range(rank):
            offset += vals[i] * stride[i]
        return offset

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
    fn _compute_tile_layout[M: Int, N: Int]() -> Layout:
        alias tiler = MakeLayoutList(Layout(M, 1), Layout(N, 1))
        return zipped_divide(layout, tiler)

    fn tile[
        M1: Int,
        N1: Int,
        *,
        __tiled_layout: Layout = Self._compute_tile_layout[M1, N1](),
    ](self, m: Int, n: Int) -> LayoutTensor[
        __tiled_layout[0], dtype, address_space
    ]:
        alias tiled_layout_stride = Self._toStatic[__tiled_layout[1].stride]()
        var offset = m * tiled_layout_stride[0] + n * tiled_layout_stride[1]
        return LayoutTensor[__tiled_layout[0], dtype, address_space](
            self.ptr.offset(offset)
        )

    @staticmethod
    fn _compute_distribute_layout[
        data_layout: Layout, threads_layout: Layout
    ]() -> Layout:
        var thread_tile = LayoutList()
        for dim in threads_layout.shape:
            thread_tile.append(Layout(dim))
        return zipped_divide(layout, thread_tile)

    fn distribute[
        threads_layout: Layout,
        tiled_layout: Layout = Self._compute_distribute_layout[
            layout, threads_layout
        ](),
    ](self, m: Int, n: Int) -> LayoutTensor[
        tiled_layout[1], dtype, address_space
    ]:
        alias composed_layout = composition(tiled_layout[0], threads_layout)
        alias fragments_layout_stride = Self._toStatic[composed_layout.stride]()
        var offset = m * fragments_layout_stride[
            0
        ] + n * fragments_layout_stride[1]
        return LayoutTensor[tiled_layout[1], dtype, address_space](
            self.ptr.offset(offset)
        )

    fn transpose[
        M: Int = Self.dim[0](),
        N: Int = Self.dim[1](),
        transposed_layout: Layout = composition(
            layout,
            Layout(IntTuple(N, M), IntTuple(M, 1)),
        ),
    ](self) -> LayoutTensor[transposed_layout, dtype, address_space]:
        return LayoutTensor[transposed_layout, dtype, address_space](self.ptr)

    fn reshape[
        dst_layout: Layout,
        reshaped_layout: Layout = composition(layout, dst_layout),
    ](self) -> LayoutTensor[reshaped_layout, dtype, address_space]:
        return LayoutTensor[reshaped_layout, dtype, address_space](self.ptr)

    @always_inline
    fn copy_from[
        other_layout: Layout
    ](self, other: LayoutTensor[other_layout, dtype, address_space]):
        for m in range(Self.dim[0]()):

            @parameter
            if (
                int(self.layout.stride[1]) <= 1
                and int(other.layout.stride[1]) <= 1
                and not triple_is_nvidia_cuda()
            ):
                # Optimize copy for row major layouts.
                memcpy(
                    self.ptr.offset(self._offset(m, 0)),
                    other.ptr.offset(other._offset(m, 0)),
                    Self.dim[1](),
                )
            else:
                for n in range(Self.dim[1]()):
                    self[m, n] = other[m, n]

    # When source and destination address spaces differ
    @always_inline
    fn copy_from_numa[
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


struct TensorBuilder[
    M: Int,
    N: Int,
    dtype: DType,
    layout: Layout = Layout(IntTuple(M, N), IntTuple(N, 1)),
]:
    alias Type = LayoutTensor[layout, dtype]
    alias OwningType = LayoutTensor[layout, dtype, owning=True]

    @staticmethod
    fn Wrap(ptr: DTypePointer[dtype]) -> Self.Type:
        return Self.Type(ptr)

    @staticmethod
    fn Build() -> Self.OwningType:
        return Self.OwningType(DTypePointer[dtype].alloc(M * N))

    @staticmethod
    fn _aligned_layout() -> Layout:
        alias alignment = alignof[SIMD[dtype]]()
        alias n_aligned = ((N + alignment - 1) // alignment) * alignment
        alias data_layout = Layout(
            IntTuple(M, n_aligned), IntTuple(n_aligned, 1)
        )
        return LayoutTensor[data_layout, dtype]._compute_tile_layout[M, N]()[0]

    @staticmethod
    fn BuildAligned[
        *, __target_layout: Layout = Self._aligned_layout()
    ]() -> LayoutTensor[__target_layout, dtype, owning=True]:
        var ptr = DTypePointer[dtype].alloc(
            M * int(__target_layout.stride[0]), alignment=alignof[SIMD[dtype]]()
        )
        return LayoutTensor[__target_layout, dtype, owning=True](ptr)


fn stack_allocation_like[
    layout: Layout,
    dtype: DType,
    address_space: AddressSpace,
    target_address_space: AddressSpace = AddressSpace.GENERIC,
](in_tensor: LayoutTensor[layout, dtype, address_space]) -> LayoutTensor[
    layout, dtype, target_address_space
]:
    return LayoutTensor[layout, dtype, target_address_space].stack_allocation()
