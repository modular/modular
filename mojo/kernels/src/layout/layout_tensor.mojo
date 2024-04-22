# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections.optional import Optional, OptionalReg
from sys.info import sizeof
from sys.intrinsics import PrefetchOptions

from algorithm import vectorize
from gpu import WARP_SIZE
from gpu.memory import async_copy, async_copy_wait_all
from gpu.id import ThreadIdx
from memory import memcpy
from memory.unsafe import DTypePointer
from memory.reference import AddressSpace, _GPUAddressSpace

from .int_tuple import flatten, idx2crd, int, product, fill_like
from .layout import *
from math.limit import max_finite
from builtin.int import int as _int


# Distribute thread_layout into data_layout, if axis is provided
# distribute into threads_layout projected into this axis.
#
fn _compute_distribute_layout[
    data_layout: Layout,
    threads_layout: Layout,
    axis: Optional[Int] = None,
]() -> Layout:
    var thread_tile = LayoutList()

    @parameter
    if axis:
        return zipped_divide(
            data_layout, Layout(threads_layout.shape[axis._value_copy()])
        )

    for dim in threads_layout.shape:
        thread_tile.append(Layout(dim))
    return zipped_divide(data_layout, thread_tile)


# Returns an IntTuple with all ones except axis same as input t, when
# submode_axis is provided the projection happens on the submode only.
fn _project_on_axis[
    axis: Int, submode_axis: Optional[Int] = None
](t: IntTuple) -> IntTuple:
    if not submode_axis:
        var p_t = fill_like(t, 0)
        p_t[axis] = fill_like(t[axis], 1)
        return p_t
    var p_t = fill_like(t, 1)
    p_t[axis] = fill_like(t[axis], 0)
    p_t[axis][submode_axis._value_copy()] = 1
    return p_t


fn _get_index_type(layout: Layout, address_space: AddressSpace) -> DType:
    if layout.cosize() < _int(max_finite[DType.int32]()):
        return DType.int32
    elif (
        address_space == _GPUAddressSpace.SHARED
        or address_space == _GPUAddressSpace.CONSTANT
    ):
        return DType.int32
    else:
        return DType.index


alias _swizzle_signature = fn[type: DType] (Scalar[type]) -> Scalar[type]


@register_passable
struct LayoutTensor[
    dtype: DType,
    layout: Layout,
    rank: Int = layout.rank(),
    /,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
    element_layout: Layout = Layout(1, 1),
    index_type: DType = _get_index_type(layout, address_space),
](CollectionElement):
    var ptr: DTypePointer[dtype, address_space]
    var owning: Bool

    alias element_size = element_layout.size()
    alias element_type = SIMD[dtype, Self.element_size]

    @always_inline
    fn __init__(
        inout self,
        ptr: DTypePointer[dtype, address_space],
        /,
        *,
        owning: Bool = False,
    ):
        self.ptr = ptr
        self.owning = owning

    @always_inline
    fn __copyinit__(inout self: Self, existing: Self):
        self.ptr = existing.ptr
        self.owning = False

    fn __del__(owned self):
        @parameter
        if triple_is_nvidia_cuda() and (
            address_space == _GPUAddressSpace.GENERIC
            or address_space == _GPUAddressSpace.GLOBAL
        ):
            # Owned tensors live in GENERIC address space only futheremore you
            # can only allocate in GENERIC address space, so can skip the free.
            if self.owning:
                self.ptr.free()

    @always_inline
    fn _offset(self, m: Int, n: Int) -> Int:
        return Self.stride[0]() * m + Self.stride[1]() * n

    @always_inline
    fn __getitem__(self, *dims: Int) -> Self.element_type:
        # FIXME: Enable debug_assert, now fails with INVALID_PTX
        # debug_assert(
        #     dims.__len__() == Self.rank(),
        #     "getitem should have same number of indices as the rank",
        # )
        alias strides = Self._toStatic[flatten(layout.stride)]()
        var vec_res = SIMD[dtype, Self.element_size]()

        # TODO: We should vectorize the reads of contiguous loads this just stash
        # scalar elements.
        @parameter
        fn fill_vec_res[idx: Int]():
            alias element_offset = self.element_layout(idx)
            vec_res[idx] = self.ptr.load(
                Self._getOffset(strides, dims) + element_offset
            )

        unroll[fill_vec_res, Self.element_size]()

        return vec_res

    @always_inline
    fn __setitem__(self, d0: Int, val: Self.element_type):
        alias strides = Self._toStatic[flatten(layout.stride)]()

        # TODO: We should vectorize contiguous stores this just stash scalars.
        @parameter
        fn store_element[i: Int]():
            alias element_offset = self.element_layout(i)
            self.ptr.store(
                Self._getOffset(strides, VariadicList[Int](d0))
                + element_offset,
                val[i],
            )

        unroll[store_element, Self.element_size]()

    @always_inline
    fn __setitem__(self, d0: Int, d1: Int, val: Self.element_type):
        alias strides = Self._toStatic[flatten(layout.stride)]()

        @parameter
        fn store_element[i: Int]():
            alias element_offset = self.element_layout(i)
            self.ptr.store(
                Self._getOffset(strides, VariadicList[Int](d0, d1))
                + element_offset,
                val[i],
            )

        unroll[store_element, Self.element_size]()

    @always_inline
    fn __setitem__(self, d0: Int, d1: Int, d2: Int, val: Self.element_type):
        alias strides = Self._toStatic[flatten(layout.stride)]()

        @parameter
        fn store_element[i: Int]():
            alias element_offset = self.element_layout(i)
            self.ptr.store(
                Self._getOffset(strides, VariadicList[Int](d0, d1, d2))
                + element_offset,
                val[i],
            )

        unroll[store_element, Self.element_size]()

    @always_inline
    fn __setitem__(
        self, d0: Int, d1: Int, d2: Int, d3: Int, val: Self.element_type
    ):
        alias strides = Self._toStatic[flatten(layout.stride)]()

        @parameter
        fn store_element[i: Int]():
            alias element_offset = self.element_layout(i)
            self.ptr.store(
                Self._getOffset(strides, VariadicList[Int](d0, d1, d2, d3))
                + element_offset,
                val[i],
            )

        unroll[store_element, Self.element_size]()

    @always_inline
    fn load[width: Int](self, m: Int, n: Int) -> SIMD[dtype, width]:
        return self.ptr.load[width=width](self._offset(m, n))

    @always_inline
    fn prefetch(self, m: Int, n: Int):
        self.ptr.offset(self._offset(m, n)).prefetch[
            PrefetchOptions().for_read().high_locality().to_data_cache()
        ]()

    @always_inline
    fn aligned_load[width: Int](self, m: Int, n: Int) -> SIMD[dtype, width]:
        alias alignment = alignof[SIMD[dtype, width]]()
        return self.ptr.load[width=width, alignment=alignment](
            self._offset(m, n)
        )

    @always_inline
    fn store[width: Int](self, m: Int, n: Int, val: SIMD[dtype, width]):
        return self.ptr.store[width=width](self._offset(m, n), val)

    @always_inline
    fn aligned_store[width: Int](self, m: Int, n: Int, val: SIMD[dtype, width]):
        alias alignment = alignof[SIMD[dtype, width]]()
        return self.ptr.store[width=width, alignment=alignment](
            self._offset(m, n), val
        )

    @staticmethod
    @always_inline("nodebug")
    fn stack_allocation() -> Self:
        return stack_allocation[
            layout.size(), dtype, address_space=address_space
        ]()

    @staticmethod
    @always_inline("nodebug")
    fn aligned_stack_allocation[alignment: Int]() -> Self:
        return stack_allocation[
            layout.size(),
            dtype,
            alignment=alignment,
            address_space=address_space,
        ]()

    @staticmethod
    @always_inline("nodebug")
    fn _toStatic[t: IntTuple]() -> StaticIntTuple[len(t)]:
        var st = StaticIntTuple[len(t)]()

        @unroll
        for i in range(len(t)):
            st[i] = int(t[i])
        return st

    @staticmethod
    @always_inline("nodebug")
    fn _getOffset[
        rank: Int
    ](stride: StaticIntTuple[rank], vals: VariadicList[Int]) -> Int:
        var offset = 0

        @unroll
        for i in range(rank):
            offset += vals[i] * stride[i]
        return offset

    @staticmethod
    @always_inline("nodebug")
    fn _getOffset[
        rank_1: Int, rank_2: Int
    ](stride: StaticIntTuple[rank_1], vals: StaticIntTuple[rank_2]) -> Int:
        # In theory we should be able to verify this at compile time but it not happening now!
        constrained[
            rank_1 == rank_2, "shape and stride should be the same rank!"
        ]()
        var offset = 0

        @unroll
        for i in range(rank_1):
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

    @always_inline
    fn coalesce(
        self,
    ) -> LayoutTensor[
        dtype,
        coalesce(layout),
        address_space=address_space,
        element_layout = self.element_layout,
    ]:
        return LayoutTensor[
            dtype,
            coalesce(layout),
            address_space=address_space,
            element_layout = self.element_layout,
        ](self.ptr)

    @staticmethod
    fn _compute_tile_layout[*tile_sizes: Int]() -> Layout:
        alias tiler = MakeTileLayoutList[tile_sizes]()
        return zipped_divide(layout, tiler)

    @staticmethod
    fn _compute_tile_layout[tile_size: Int, axis: Int]() -> Layout:
        var tiler = LayoutList()
        var i = 0
        for dim in layout.shape:
            if i == axis:
                tiler.append(Layout(tile_size))
            else:
                tiler.append(Layout(dim))
            i += 1
        return zipped_divide(layout, tiler)

    @always_inline
    fn tile[
        *tile_sizes: Int,
        __tiled_layout: Layout = Self._compute_tile_layout[tile_sizes](),
    ](self, *tile_coords: Int) -> LayoutTensor[
        dtype, __tiled_layout[0], address_space=address_space
    ]:
        @parameter
        fn num_tiles() -> Int:
            return __mlir_op.`pop.variadic.size`(tile_sizes)

        constrained[
            __tiled_layout[1].rank() == num_tiles(),
            "Number of tiles should match the rank",
        ]()

        var offset = 0

        @parameter
        fn compute_offset[i: Int]():
            alias stride = int(__tiled_layout[1].stride[i])
            offset += tile_coords[i] * stride

        unroll[compute_offset, num_tiles()]()

        return LayoutTensor[
            dtype, __tiled_layout[0], address_space=address_space
        ](self.ptr.offset(offset))

    @always_inline
    fn split[
        count: Int,
        axis: Int = 0,
        __tile_size: Int = layout.shape[axis].value() // count,
        __tiled_layout: Layout = Self._compute_tile_layout[__tile_size, axis](),
    ](self) -> StaticTuple[
        LayoutTensor[dtype, __tiled_layout[0], address_space=address_space],
        count,
    ]:
        constrained[
            layout.shape[axis].is_value(),
            "Only support partition modes that are plain values.",
        ]()

        constrained[
            layout.shape[axis].value() % count == 0,
            "The input dimension must be divisible over the input count.",
        ]()

        alias stride = layout.stride[axis].value()

        var tiles = StaticTuple[
            LayoutTensor[dtype, __tiled_layout[0], address_space=address_space],
            count,
        ]()

        @unroll
        for i in range(count):
            tiles[i] = LayoutTensor[
                dtype, __tiled_layout[0], address_space=address_space
            ](self.ptr.offset(i * __tile_size * stride))

        return tiles

    @staticmethod
    fn _compute_distribute_layout[
        data_layout: Layout,
        threads_layout: Layout,
        axis: Optional[Int] = None,
    ]() -> Layout:
        var thread_tile = LayoutList()

        @parameter
        if axis:
            return zipped_divide(
                data_layout, Layout(threads_layout.shape[axis._value_copy()])
            )
        else:
            for dim in threads_layout.shape:
                thread_tile.append(Layout(dim))

            return zipped_divide(data_layout, thread_tile)

    @always_inline
    fn distribute[
        threads_layout: Layout,
        axis: Optional[Int] = None,
        swizzle: OptionalReg[_swizzle_signature] = None,
        tiled_layout: Layout = _compute_distribute_layout[
            layout, threads_layout, axis
        ](),
        submode_axis: Optional[Int] = None,
    ](self, thread_id: Int32) -> LayoutTensor[
        dtype,
        tiled_layout[1],
        address_space=address_space,
        element_layout=element_layout,
    ]:
        """Distribute tiled workload to threads.

        If the `axis` is given, for example, using `axis = 0` for 4 threads:
        TH_0 TH_2
        TH_1 TH_3
        This means the tensor is only distributed to threads in axis = 0, i.e.,
        threads 0 and 1. Threads 2 and 3 gets the same tile as 0 and 1, respectively.
        This is useful when threads load same vectors from a row in A matrix and
        some threads share the same vector.
        """
        alias fragments_layout_stride = flatten(tiled_layout[0].stride)

        alias threads_layout_shape = flatten(threads_layout.shape)
        alias threads_layout_stride = flatten(threads_layout.stride)

        # Only extract coordinates in the given axis.
        # Example: axis = 0 for 2x2 threads, we only need thread 0 and 1's
        # coordinates since thread 2 and 3 are getting the same tile.
        alias thread_projected_stride = flatten(
            threads_layout.stride[
                axis._value_copy()
            ] if axis else threads_layout.stride
        )
        alias thread_projected_shape = flatten(
            threads_layout.shape[
                axis._value_copy()
            ] if axis else threads_layout.shape
        )

        var offset: Scalar[Self.index_type] = 0

        @parameter
        fn compute_offset[i: Int]():
            alias fragments_stride_i = int(fragments_layout_stride[i])
            alias shape_i = int(thread_projected_shape[i])
            alias stride_i = int(thread_projected_stride[i])
            var thread_coord_i = (thread_id // stride_i) % shape_i
            offset += (
                thread_coord_i.cast[Self.index_type]() * fragments_stride_i
            )

        unroll[compute_offset, len(fragments_layout_stride)]()

        # Swizzling applies to the index of elements rather than scalars because
        # the former is the unit in distribution.
        var swizzled_offset = offset

        @parameter
        if swizzle:
            alias swizzle_fn = swizzle.value()
            swizzled_offset = (
                swizzle_fn[Self.index_type](offset // self.element_size)
                * self.element_size
            )

        return LayoutTensor[
            dtype,
            tiled_layout[1],
            address_space=address_space,
            element_layout=element_layout,
        ](self.ptr.offset(swizzled_offset))

    @always_inline
    fn vectorize[
        *tile_sizes: Int,
        __tiled_layout: Layout = Self._compute_tile_layout[tile_sizes](),
    ](self) -> LayoutTensor[
        dtype,
        coalesce(__tiled_layout[1], keep_rank=True),
        address_space=address_space,
        element_layout = coalesce(__tiled_layout[0]),
    ]:
        return LayoutTensor[
            dtype,
            coalesce(__tiled_layout[1], keep_rank=True),
            address_space=address_space,
            element_layout = coalesce(__tiled_layout[0]),
        ](self.ptr)

    @staticmethod
    fn __get_slice_size(slice: Slice, dim: Int) -> Int:
        var end = slice.end if slice._has_end() else int(Self.layout.shape[dim])
        return end - slice.start

    @staticmethod
    fn __compute_slice_layout(d0_slice: Slice, d1_slice: Slice) -> Layout:
        constrained[
            layout.shape.__len__() == 2,
            "Only rank-2 tensors slices are supported for now!",
        ]()
        return Layout(
            IntTuple(
                Self.__get_slice_size(d0_slice, 0),
                Self.__get_slice_size(d1_slice, 1),
            ),
            layout.stride,
        )

    @always_inline
    fn slice[
        d0_slice: Slice,
        d1_slice: Slice,
        __slice_layout: Layout = Self.__compute_slice_layout(
            d0_slice,
            d1_slice,
        ),
    ](self) -> LayoutTensor[
        dtype,
        __slice_layout,
        address_space=address_space,
        element_layout=element_layout,
    ]:
        constrained[
            d0_slice.step == 1 and d1_slice.step == 1,
            "Slice should have no gaps",
        ]()
        alias stride_m = int(__slice_layout.stride[0])
        alias stride_n = int(__slice_layout.stride[1])
        var offset = d0_slice.start * stride_m + d1_slice.start * stride_n
        return LayoutTensor[
            dtype,
            __slice_layout,
            address_space=address_space,
            element_layout=element_layout,
        ](self.ptr.offset(offset))

    @always_inline
    fn transpose[
        M: Int = Self.dim[0](),
        N: Int = Self.dim[1](),
        transposed_layout: Layout = composition(
            layout,
            Layout(IntTuple(N, M), IntTuple(M, 1)),
        ),
    ](self) -> LayoutTensor[
        dtype,
        transposed_layout,
        address_space=address_space,
        element_layout=element_layout,
    ]:
        return LayoutTensor[
            dtype,
            transposed_layout,
            address_space=address_space,
            element_layout=element_layout,
        ](self.ptr)

    @always_inline
    fn reshape[
        dst_layout: Layout,
        reshaped_layout: Layout = composition(layout, dst_layout),
    ](self) -> LayoutTensor[
        dtype,
        reshaped_layout,
        address_space=address_space,
        element_layout=element_layout,
    ]:
        return LayoutTensor[
            dtype,
            reshaped_layout,
            address_space=address_space,
            element_layout=element_layout,
        ](self.ptr)

    @always_inline
    fn copy_from[
        other_layout: Layout
    ](
        self,
        other: LayoutTensor[
            dtype,
            other_layout,
            address_space=address_space,
            element_layout=element_layout,  # TODO: Remove this assumtion.
        ],
    ):
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
        other_layout: Layout,
        other_addr_space: AddressSpace,
        other_element_layout: Layout,
    ](
        self,
        other: LayoutTensor[
            dtype,
            other_layout,
            address_space=other_addr_space,
            element_layout=other_element_layout,
        ],
    ):
        alias dst_size = layout.size()
        alias src_size = other_layout.size()

        alias dst_element_size = int(self.element_size)
        alias src_element_size = int(other.element_size)

        constrained[
            dst_size == src_size, "copy_from should move data of the same size"
        ]()

        constrained[
            dst_element_size == src_element_size, "copy_from should move"
        ]()

        # alias align = alignof[SIMD[dtype, self.element_size]]()

        # Vectorize 1-D element read/writes.
        @parameter
        if (
            other_element_layout.rank() == 1
            and other_element_layout.stride[0] == 1
            and self.element_layout.rank() == 1
            and self.element_layout.stride[0] == 1
        ):

            @parameter
            fn copy_vector_to_vector[i: Int]():
                alias src_idx = make_layout(other.element_layout, other_layout)(
                    i * src_element_size
                )
                alias dst_idx = make_layout(self.element_layout, self.layout)(
                    i * dst_element_size
                )
                var src_vec = rebind[self.element_type](
                    other.ptr.load[
                        width=src_element_size,
                        alignment = alignof[other.element_type](),
                    ](src_idx)
                )
                self.ptr.store[
                    width = self.element_size,
                    alignment = alignof[self.element_type](),
                ](dst_idx, src_vec)

            unroll[copy_vector_to_vector, dst_size]()
        # Vector read scalar writes.
        elif (
            other_element_layout.rank() == 1
            and other_element_layout.stride[0] == 1
            and self.element_layout.rank() == 1
            and self.element_layout.stride[0] != 1
        ):

            @parameter
            fn copy_vector_to_scalars[i: Int]():
                alias src_idx = make_layout(other.element_layout, other_layout)(
                    i * src_element_size
                )

                var src_vec = rebind[self.element_type](
                    other.ptr.load[
                        width=src_element_size,
                        alignment = alignof[self.element_type](),
                    ](src_idx)
                )

                @parameter
                fn store_vector[e_i: Int]():
                    alias dst_idx = make_layout(
                        self.element_layout, self.layout
                    )(i * dst_element_size + e_i)
                    self.ptr.store[width=1](dst_idx, src_vec[e_i])

                unroll[store_vector, src_element_size]()

            unroll[copy_vector_to_scalars, src_size]()
        # Vector write scalar reads.
        elif (
            other_element_layout.rank() == 1
            and other_element_layout.stride[0] != 1
            and self.element_layout.rank() == 1
            and self.element_layout.stride[0] == 1
        ):

            @parameter
            fn copy_scalars_to_vectors[i: Int]():
                alias dst_idx = make_layout(self.element_layout, self.layout)(
                    i * dst_element_size
                )

                var src_vec = self.element_type()

                @parameter
                fn fill_vector[e_i: Int]():
                    alias src_idx = make_layout(
                        other_element_layout, other.layout
                    )(i * src_element_size + e_i)
                    src_vec[e_i] = other.ptr.load[width=1](src_idx)

                unroll[fill_vector, src_element_size]()
                self.ptr.store[width = self.element_size](dst_idx, src_vec)

            unroll[copy_scalars_to_vectors, dst_size]()
        # Vectorized copy between 2D row-major elements, used for C in gemm.
        elif (
            # Not trivial element
            self.element_layout != Layout(IntTuple(1, 1))
            and self.element_layout.rank() == 2
            and other_element_layout.shape == self.element_layout.shape
            and other_element_layout.stride[1] == 1
            and self.element_layout.stride[1] == 1
        ):
            # Copy an element tensor.
            @parameter
            fn copy_by_element[i: Int]():
                # Offset to the current element.
                alias src_offset = other_layout(i)
                alias dst_offset = self.layout(i)
                alias num_copies = self.element_layout.shape[0].value()
                alias vec_width = self.element_layout.shape[1].value()

                @parameter
                fn copy_by_vec[j: Int]():
                    alias src_idx = src_offset + other_element_layout(j)
                    alias dst_idx = dst_offset + self.element_layout(j)

                    var src_vec = other.ptr.load[
                        width=vec_width,
                        alignment = alignof[SIMD[dtype, vec_width]](),
                    ](src_idx).cast[dtype]()

                    self.ptr.store[
                        width=vec_width,
                        alignment = alignof[SIMD[dtype, vec_width]](),
                    ](dst_idx, src_vec)

                unroll[copy_by_vec, num_copies]()

            unroll[copy_by_element, dst_size]()

        else:

            @parameter
            fn copy_element[i: Int]():
                alias src_idx = make_layout(other.element_layout, other_layout)(
                    i
                )
                alias dst_idx = make_layout(self.element_layout, self.layout)(i)

                self.ptr[dst_idx] = other.ptr[src_idx]

            unroll[copy_element, dst_size * dst_element_size]()

    @always_inline
    fn copy_from_async[
        src_layout: Layout,
        src_addr_space: AddressSpace,
        src_element_layout: Layout,
    ](
        self,
        src: LayoutTensor[
            dtype,
            src_layout,
            address_space=src_addr_space,
            element_layout=src_element_layout,
        ],
    ):
        constrained[
            self.address_space == _GPUAddressSpace.SHARED,
            "Async is only supported for destinations in shared memory",
        ]()

        alias dst_size = layout.size()
        alias src_size = src_layout.size()
        constrained[
            dst_size == src_size,
            "copy_from_async should move data of the same size",
        ]()

        alias dst_element_size = int(self.element_size)
        alias src_element_size = int(src.element_size)
        constrained[
            dst_element_size == src_element_size,
            "copy_from_async should move data of the same element size",
        ]()

        # Eligibility for 4, 8, 16 bytes async load.
        alias element_size_bytes = sizeof[dtype]() * src_element_size
        constrained[
            element_size_bytes == 4
            or element_size_bytes == 8
            or element_size_bytes == 16,
            "copy_from_async only allows 4, 8, 16 bytes element",
        ]()

        var dst_ptr = self.ptr.bitcast[
            address_space = _GPUAddressSpace.SHARED
        ]()
        var src_ptr = src.ptr.bitcast[address_space = _GPUAddressSpace.GLOBAL]()

        @parameter
        if (
            src_element_layout.rank() == 1
            and src_element_layout.stride[0] == 1
            and self.element_layout.rank() == 1
            and self.element_layout.stride[0] == 1
        ):

            @parameter
            fn copy_vector_to_vector[i: Int]():
                alias src_idx = make_layout(src.element_layout, src_layout)(
                    i * src_element_size
                )
                alias dst_idx = make_layout(self.element_layout, self.layout)(
                    i * dst_element_size
                )

                async_copy[element_size_bytes](
                    src_ptr + src_idx, dst_ptr + dst_idx
                )

            unroll[copy_vector_to_vector, dst_size]()

        else:

            @parameter
            fn copy_element[i: Int]():
                alias src_idx = make_layout(src.element_layout, src_layout)(i)
                alias dst_idx = make_layout(self.element_layout, self.layout)(i)

                async_copy[4](src_ptr + src_idx, dst_ptr + dst_idx)

            unroll[copy_element, dst_size * dst_element_size]()

    fn linspace(self):
        @parameter
        if len(layout) == 1:
            for m in range(Self.dim[0]()):
                self.ptr[m] = m

        elif len(layout) == 2:
            for m in range(Self.dim[0]()):
                for n in range(Self.dim[1]()):
                    self[m, n] = m * Self.dim[1]() + n
        else:
            abort("LayoutTensor linspace only support rank 1-2 layouts.")

    @always_inline
    fn fill(self, val: Scalar[dtype]):
        alias num_elements = layout.size() * Self.element_size

        @unroll
        for i in range(num_elements):
            self.ptr[i] = val

    fn print(self):
        """Print 2D tensor in 2D, otherwise print all values in column major
        coordinate order."""

        @always_inline
        fn is_2d_print(layout: Layout) -> Bool:
            return (
                len(layout) == 2
                and layout.shape[0].is_value()
                and layout.shape[1].is_value()
            )

        # The 2D print works only for layout shape (M, N).
        # Check both original and coalesced layouts so that (M, 1) and
        # ((M), (N)) can all be printed in 2D. Shapes like ((2, 2), 2) will be
        # printed elementwise.
        @parameter
        if is_2d_print(layout) or is_2d_print(coalesce(layout)):
            for m in range(Self.dim[0]()):
                for n in range(Self.dim[1]()):
                    print(self[m, n], end=" ")
                print("")
        else:
            for i in range(layout.size()):
                var vec_offset = layout(i)
                var vec = SIMD[dtype, Self.element_size]()

                @parameter
                fn fill_vec[idx: Int]():
                    alias element_offset = self.element_layout(idx)
                    vec[idx] = self.ptr.load[width=1](
                        vec_offset + element_offset
                    )

                unroll[fill_vec, Self.element_size]()
                print(vec)


struct TensorBuilder[
    M: Int,
    N: Int,
    dtype: DType,
    address_space: AddressSpace = AddressSpace.GENERIC,
    layout: Layout = Layout(IntTuple(M, N), IntTuple(N, 1)),
]:
    alias Type = LayoutTensor[dtype, layout, address_space=address_space]
    alias AlignedType = LayoutTensor[dtype, Self._aligned_layout()]

    @staticmethod
    fn Wrap(ptr: DTypePointer[dtype, address_space]) -> Self.Type:
        return Self.Type(ptr)

    @staticmethod
    fn Build() -> Self.Type:
        return Self.Type(
            DTypePointer[dtype, address_space].alloc(
                M * N, alignment=alignof[SIMD[dtype]]()
            ),
            owning=True,
        )

    @staticmethod
    fn OnStack() -> Self.Type:
        return Self.Type.stack_allocation()

    @staticmethod
    fn OnStackAligned[alignment: Int]() -> Self.Type:
        return Self.Type.aligned_stack_allocation[alignment]()

    @staticmethod
    fn _aligned_layout() -> Layout:
        alias alignment = alignof[SIMD[dtype]]()
        alias n_aligned = ((N + alignment - 1) // alignment) * alignment
        alias data_layout = Layout(
            IntTuple(M, n_aligned), IntTuple(n_aligned, 1)
        )
        return LayoutTensor[dtype, data_layout]._compute_tile_layout[M, N]()[0]

    @staticmethod
    fn BuildAligned[
        *, __target_layout: Layout = Self._aligned_layout()
    ]() -> LayoutTensor[dtype, __target_layout]:
        var ptr = DTypePointer[dtype].alloc(
            M * int(__target_layout.stride[0]), alignment=alignof[SIMD[dtype]]()
        )
        return LayoutTensor[dtype, __target_layout](ptr, owning=True)


fn stack_allocation_like[
    layout: Layout,
    dtype: DType,
    *,
    address_space: AddressSpace,
    target_address_space: AddressSpace = AddressSpace.GENERIC,
](
    in_tensor: LayoutTensor[dtype, layout, address_space=address_space]
) -> LayoutTensor[dtype, layout, address_space=target_address_space]:
    return LayoutTensor[
        dtype, layout, address_space=target_address_space
    ].stack_allocation()


# Updates res with the outer product of lhs, rhs vectors, res += outer(lhs, rhs).
#
@always_inline
fn outer_product_acc[
    dtype: DType,
    *,
    res_address_space: AddressSpace,
    lhs_address_space: AddressSpace,
    rhs_address_space: AddressSpace,
    res_layout: Layout,
    lhs_layout: Layout,
    rhs_layout: Layout,
](
    res: LayoutTensor[dtype, res_layout, address_space=res_address_space],
    lhs: LayoutTensor[_, lhs_layout, address_space=lhs_address_space],
    rhs: LayoutTensor[_, rhs_layout, address_space=rhs_address_space],
):
    constrained[res.rank == 2, "Only rank 2 res is allowed."]()
    constrained[lhs.rank == 1, "Only rank 1 lhs is allowed."]()
    constrained[rhs.rank == 1, "Only rank 1 rhs is allowed."]()

    alias M = res.shape[0]()
    alias N = res.shape[1]()

    constrained[lhs.shape[0]() == M, "lhs shape mismatch"]()
    constrained[rhs.shape[0]() == N, "rhs shape mismatch"]()

    @unroll
    for i in range(M):

        @unroll
        for j in range(N):
            res[i, j] += lhs[i].cast[dtype]() * rhs[j].cast[dtype]()


# Copy from DRAM -> SRAM, this requires w/r thread affinity mapping.
#
@always_inline
fn copy_dram_to_sram_async[
    src_layout: Layout,
    dst_layout: Layout,
    dtype: DType,
    src_thread_layout: Layout,
    dst_thread_layout: Layout,
    src_element_layout: Layout,
    dst_element_layout: Layout,
    swizzle: OptionalReg[_swizzle_signature] = None,
](
    dst: LayoutTensor[
        dtype,
        dst_layout,
        address_space = _GPUAddressSpace.SHARED,
        element_layout=dst_element_layout,
    ],
    src: LayoutTensor[
        dtype,
        src_layout,
        address_space = _GPUAddressSpace.GENERIC,
        element_layout=src_element_layout,
    ],
):
    var src_framgents = src.distribute[src_thread_layout](ThreadIdx.x())
    var dst_framgents = dst.distribute[dst_thread_layout, swizzle=swizzle](
        ThreadIdx.x()
    )
    dst_framgents.copy_from_async(src_framgents)


# Copy from SRAM to local memory.
#
@always_inline
fn copy_sram_to_local[
    src_layout: Layout,
    dst_layout: Layout,
    dtype: DType,
    src_warp_layout: Layout,
    src_element_layout: Layout,
    dst_element_layout: Layout,
    axis: Optional[Int] = None,
](
    dst: LayoutTensor[
        dtype,
        dst_layout,
        address_space = _GPUAddressSpace.GENERIC,
        element_layout=dst_element_layout,
    ],
    src: LayoutTensor[
        dtype,
        src_layout,
        address_space = _GPUAddressSpace.SHARED,
        element_layout=src_element_layout,
    ],
):
    @parameter
    if axis:
        var src_fragments = src.distribute[
            src_warp_layout, axis = axis._value_copy()
        ](ThreadIdx.x())
        dst.copy_from_numa(src_fragments)
    else:
        var src_fragments = src.distribute[src_warp_layout](ThreadIdx.x())
        dst.copy_from_numa(src_fragments)


# Copy local memory to DRAM, thread affinity is needed only for dst fragments.
#
@always_inline
fn copy_local_to_dram[
    src_layout: Layout,
    dst_layout: Layout,
    dtype: DType,
    dst_thread_layout: Layout,
    src_element_layout: Layout,
    dst_element_layout: Layout,
](
    dst: LayoutTensor[
        dtype,
        dst_layout,
        address_space = _GPUAddressSpace.GENERIC,
        element_layout=dst_element_layout,
    ],
    src: LayoutTensor[
        dtype,
        src_layout,
        address_space = _GPUAddressSpace.GENERIC,
        element_layout=src_element_layout,
    ],
):
    var dst_framgents = dst.distribute[dst_thread_layout](ThreadIdx.x())
    dst_framgents.copy_from_numa(src)
