# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections.optional import Optional
from sys.info import sizeof
from sys.intrinsics import PrefetchOptions

from algorithm import vectorize
from gpu.memory import async_copy, async_copy_wait_all
from memory import memcpy
from memory.unsafe import AddressSpace, DTypePointer, _GPUAddressSpace

from .int_tuple import flatten, idx2crd, int, product, fill_like
from .layout import *


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
        for i in range(len(threads_layout.shape)):
            var shape_i = threads_layout.shape[i]
            if i == axis.value():
                thread_tile.append(Layout(shape_i))
            else:
                thread_tile.append(Layout(1))

        return zipped_divide(data_layout, thread_tile)

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
    p_t[axis][submode_axis.value()] = 1
    return p_t


@register_passable
struct LayoutTensor[
    layout: Layout,
    dtype: DType,
    /,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
    element_layout: Layout = Layout(1, 1),
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
            vec_res[idx] = self.ptr.load[width=1](
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
    fn rank() -> Int:
        return layout.shape.__len__()

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
        coalesce(layout),
        dtype,
        address_space=address_space,
        element_layout = self.element_layout,
    ]:
        return LayoutTensor[
            coalesce(layout),
            dtype,
            address_space=address_space,
            element_layout = self.element_layout,
        ](self.ptr)

    @staticmethod
    fn _compute_tile_layout[*tile_sizes: Int]() -> Layout:
        alias tiler = MakeTileLayoutList[tile_sizes]()
        return zipped_divide(layout, tiler)

    @staticmethod
    fn _compute_tile_layout[tile_sizes: IntTuple]() -> Layout:
        var tiler = LayoutList()
        for tile_size in tile_sizes:
            tiler.append(Layout(tile_size))
        return zipped_divide(layout, tiler)

    @always_inline
    fn tile[
        M1: Int,
        N1: Int,
        *,
        __tiled_layout: Layout = Self._compute_tile_layout[M1, N1](),
    ](self, m: Int, n: Int) -> LayoutTensor[
        __tiled_layout[0], dtype, address_space=address_space
    ]:
        alias stride_m = int(__tiled_layout[1].stride[0])
        alias stride_n = int(__tiled_layout[1].stride[1])
        var offset = m * stride_m + n * stride_n
        return LayoutTensor[
            __tiled_layout[0], dtype, address_space=address_space
        ](self.ptr.offset(offset))

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
                data_layout, Layout(threads_layout.shape[axis.value()])
            )
        else:
            for dim in threads_layout.shape:
                thread_tile.append(Layout(dim))

            return zipped_divide(data_layout, thread_tile)

    @always_inline
    fn distribute[
        threads_layout: Layout,
        tiled_layout: Layout = _compute_distribute_layout[
            layout, threads_layout
        ](),
        axis: Optional[Int] = None,
        submode_axis: Optional[Int] = None,
    ](self, thread_id: Int) -> LayoutTensor[
        tiled_layout[1],
        dtype,
        address_space=address_space,
        element_layout=element_layout,
    ]:
        alias fragments_layout_stride = flatten(tiled_layout[0].stride)

        alias threads_layout_shape = flatten(threads_layout.shape)
        alias threads_layout_stride = flatten(threads_layout.stride)

        # Selected projection axes 0-1 constant.
        alias projection_const = flatten(
            _project_on_axis[axis.value(), submode_axis](
                threads_layout.shape
            ) if axis else fill_like(threads_layout.shape, 1)
        )

        var offset = 0

        @parameter
        fn compute_offset[i: Int]():
            alias p_axis = int(projection_const[i])
            alias shape_i = int(threads_layout_shape[i])
            alias stride_i = int(threads_layout_stride[i])
            var coords_i = (thread_id // stride_i) % shape_i
            alias fragments_stride_i = int(fragments_layout_stride[i])
            offset += p_axis * coords_i * fragments_stride_i

        unroll[compute_offset, len(fragments_layout_stride)]()

        return LayoutTensor[
            tiled_layout[1],
            dtype,
            address_space=address_space,
            element_layout=element_layout,
        ](self.ptr.offset(offset))

    # Work around issue 34843
    @staticmethod
    fn _make_layout_wrapper(layout1: Layout, layout2: Layout) -> Layout:
        return make_layout(layout1, layout2)

    @always_inline
    fn distribute[
        threads_layout: Layout,
        tile_size: IntTuple,
        axis: Optional[Int] = None,
        __tiled_layout: Layout = Self._compute_tile_layout[tile_size](),
        __tile_distribution: Layout = Self._compute_distribute_layout[
            __tiled_layout[1], threads_layout, axis
        ](),
        __result_layout: Layout = Self._make_layout_wrapper(
            __tiled_layout[0], __tile_distribution[1]
        ),
    ](self, thread_id: Int) -> LayoutTensor[
        __result_layout,
        dtype,
        address_space=address_space,
    ]:
        """Distribute tiled workload to threads. Each thread gets several tiles
        of the input tile_size.

        The function first tiles the tensor by `tile_sizes`. __tiled_layout[1]
        indicates how the tile pattern gets repeated in the tensor. Then, the
        function tiles __tiled_layout[1] by the thread_layout. This effectively
        distribute the tiles to each thread.

        If the `axis` is given, for example, using `axis = 0` for 4 threads:
        TH_0 TH_2
        TH_1 TH_3
        This means the tensor is only distributed to threads in axis = 0, i.e.,
        threads 0 and 1. Threads 2 and 3 gets the same tile as 0 and 1, respectively.
        This is useful when threads load same vectors from a row in A matrix and
        some threads share the same vector.
        """

        # Fragments are the tile of tile_size. The stride is between the TH_0's
        # tile to TH_1's tile. Example:
        # +-----------+-----------+
        # | TH_0 TH_0 | TH_2 TH_2 |
        # | TH_0 TH_0 | TH_2 TH_2 |
        # |-----------+-----------+
        # | TH_1 TH_1 | TH_3 TH_3 |
        # | TH_1 TH_1 | TH_3 TH_3 |
        # |-----------+-----------+
        alias fragments_layout_stride = flatten(__tile_distribution[0].stride)

        # Only extract coordinates in the given axis.
        # Example: axis = 0 in the above diagram, we only need TH_0 and TH_1's
        # coordinates since TH_2 and TH_3 are getting the same tile.
        # fmt: off
        alias thread_projected_stride = flatten(
            threads_layout.stride[axis.value()] if axis else threads_layout.stride
        )
        alias thread_projected_shape = flatten(
            threads_layout.shape[axis.value()] if axis else threads_layout.shape
        )
        # fmt: on

        var offset = 0

        @parameter
        fn compute_offset[i: Int]():
            alias fragments_stride_i = int(fragments_layout_stride[i])
            alias shape_i = int(thread_projected_shape[i])
            alias stride_i = int(thread_projected_stride[i])
            var thread_coord_i = (thread_id // stride_i) % shape_i
            offset += thread_coord_i * fragments_stride_i

        unroll[compute_offset, len(fragments_layout_stride)]()

        return LayoutTensor[
            __result_layout,
            dtype,
            address_space=address_space,
        ](self.ptr.offset(offset))

    @always_inline
    fn vectorize[
        *tile_sizes: Int,
        __tiled_layout: Layout = Self._compute_tile_layout[tile_sizes](),
    ](self) -> LayoutTensor[
        __tiled_layout[1],
        dtype,
        address_space=address_space,
        element_layout = coalesce(__tiled_layout[0]),
    ]:
        return LayoutTensor[
            __tiled_layout[1],
            dtype,
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
        __slice_layout,
        dtype,
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
            __slice_layout,
            dtype,
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
        transposed_layout,
        dtype,
        address_space=address_space,
        element_layout=element_layout,
    ]:
        return LayoutTensor[
            transposed_layout,
            dtype,
            address_space=address_space,
            element_layout=element_layout,
        ](self.ptr)

    @always_inline
    fn reshape[
        dst_layout: Layout,
        reshaped_layout: Layout = composition(layout, dst_layout),
    ](self) -> LayoutTensor[
        reshaped_layout,
        dtype,
        address_space=address_space,
        element_layout=element_layout,
    ]:
        return LayoutTensor[
            reshaped_layout,
            dtype,
            address_space=address_space,
            element_layout=element_layout,
        ](self.ptr)

    @always_inline
    fn copy_from[
        other_layout: Layout
    ](
        self,
        other: LayoutTensor[
            other_layout,
            dtype,
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
        alignment: Int = 1,
        *,
        other_layout: Layout,
        other_addr_space: AddressSpace,
        other_element_layout: Layout,
    ](
        self,
        other: LayoutTensor[
            other_layout,
            dtype,
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
                    other.ptr.load[width=src_element_size, alignment=alignment](
                        src_idx
                    )
                )
                self.ptr.store[width = self.element_size, alignment=alignment](
                    dst_idx, src_vec
                )

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
                    other.ptr.load[width=src_element_size](src_idx)
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
            src_layout,
            dtype,
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

        var dst_ptr = self.ptr.address_space_cast[_GPUAddressSpace.SHARED]()
        var src_ptr = src.ptr.address_space_cast[_GPUAddressSpace.GLOBAL]()

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

        # The 2D print works only for layout shape (M, N).
        @parameter
        if (
            len(layout) == 2
            and layout.shape[0].is_value()
            and layout.shape[1].is_value()
        ):
            for m in range(Self.dim[0]()):
                for n in range(Self.dim[1]()):
                    print(self[m, n], end=" ")
                print("")
        else:
            for i in range(layout.size()):
                var coord = idx2crd(i, layout.shape)
                var idx = layout(coord)
                print(self.ptr[idx])


struct TensorBuilder[
    M: Int,
    N: Int,
    dtype: DType,
    address_space: AddressSpace = AddressSpace.GENERIC,
    layout: Layout = Layout(IntTuple(M, N), IntTuple(N, 1)),
]:
    alias Type = LayoutTensor[layout, dtype, address_space=address_space]
    alias AlignedType = LayoutTensor[Self._aligned_layout(), dtype]

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
        return LayoutTensor[data_layout, dtype]._compute_tile_layout[M, N]()[0]

    @staticmethod
    fn BuildAligned[
        *, __target_layout: Layout = Self._aligned_layout()
    ]() -> LayoutTensor[__target_layout, dtype]:
        var ptr = DTypePointer[dtype].alloc(
            M * int(__target_layout.stride[0]), alignment=alignof[SIMD[dtype]]()
        )
        return LayoutTensor[__target_layout, dtype](ptr, owning=True)


fn stack_allocation_like[
    layout: Layout,
    dtype: DType,
    *,
    address_space: AddressSpace,
    target_address_space: AddressSpace = AddressSpace.GENERIC,
](
    in_tensor: LayoutTensor[layout, dtype, address_space=address_space]
) -> LayoutTensor[layout, dtype, address_space=target_address_space]:
    return LayoutTensor[
        layout, dtype, address_space=target_address_space
    ].stack_allocation()


# Updates res with the outer product of lhs, rhs vectors, res += outer(lhs, rhs).
#
fn outer_product_acc[
    dtype: DType
](
    res: LayoutTensor[_, dtype],
    lhs: LayoutTensor[_, _],
    rhs: LayoutTensor[_, _],
):
    constrained[res.rank() == 2, "Only rank 2 res is allowed."]()
    constrained[lhs.rank() == 1, "Only rank 1 lhs is allowed."]()
    constrained[rhs.rank() == 1, "Only rank 1 rhs is allowed."]()

    alias M = res.shape[0]()
    alias N = res.shape[1]()

    constrained[lhs.shape[0]() == M, "lhs shape mismatch"]()
    constrained[rhs.shape[0]() == N, "rhs shape mismatch"]()

    @unroll
    for i in range(M):

        @unroll
        for j in range(N):
            res[i, j] += lhs[i].cast[dtype]() * rhs[j].cast[dtype]()
