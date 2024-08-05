# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections.optional import Optional
from sys.info import sizeof
from sys.intrinsics import PrefetchOptions

from algorithm import vectorize
from builtin.int import int as _int
from gpu.id import ThreadIdx
from gpu.memory import async_copy
from layout.element import Element
from memory import memcpy, UnsafePointer, stack_allocation
from memory.reference import AddressSpace, _GPUAddressSpace

from sys import alignof, prefetch
from utils import InlineArray, StaticIntTuple
from utils.numerics import max_finite

from .int_tuple import fill_like, flatten, idx2crd, product, to_int
from .layout import *

from .runtime_layout import (
    RuntimeLayout,
    coalesce as runtime_coalesce,
    make_layout as make_runtime_layout,
)
from .runtime_tuple import RuntimeTuple


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
            data_layout, Layout(threads_layout.shape[axis.value()])
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
    p_t[axis][submode_axis.value()] = 1
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


# Returns the size of variadic integer parameters.
#
fn __get_len[*var_int: Int]() -> Int:
    return __mlir_op.`pop.variadic.size`(var_int)


# Returns the size of the slice in layout dim.
#
fn _get_slice_size(layout: Layout, slc: Slice, dim: Int) -> Int:
    var start: Int
    var end: Int
    start, end, _ = slc.indices(to_int(layout.shape[dim]))
    return end - start


# Returns true if n isn't in `tuple`.
#
fn _not_in_tuple[n: Int, size: Int, tuple: StaticIntTuple[size]]() -> Bool:
    @parameter
    for i in range(size):

        @parameter
        if tuple[i] == n:
            return False
    return True


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
](CollectionElement, CollectionElementNew):
    var ptr: UnsafePointer[Scalar[dtype], address_space]

    var runtime_layout: RuntimeLayout[layout]

    var runtime_element_layout: RuntimeLayout[element_layout]

    alias element_size = element_layout.size()
    alias element_type = SIMD[dtype, Self.element_size]

    # An offset of the global coords.
    var org_coords_offset: StaticIntTuple[rank]
    # The stride of the global coords.
    var org_coords_stride: StaticIntTuple[rank]

    # ===------------------------------------------------------------------=== #
    # Life cycle methods
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __init__(
        inout self,
        ptr: UnsafePointer[Scalar[dtype], address_space],
        /,
        *,
        org_coords_offset: StaticIntTuple[rank] = StaticIntTuple[rank](0),
        org_coords_stride: StaticIntTuple[rank] = StaticIntTuple[rank](1),
    ):
        constrained[layout.all_dims_known(), "Layout must be fully static"]()
        self.ptr = ptr
        self.runtime_layout = RuntimeLayout[layout]()
        self.runtime_element_layout = RuntimeLayout[element_layout]()
        self.org_coords_offset = org_coords_offset
        self.org_coords_stride = org_coords_stride

    @always_inline
    fn __init__(
        inout self,
        ptr: UnsafePointer[Scalar[dtype], address_space],
        runtime_layout: RuntimeLayout[layout],
        /,
        *,
        org_coords_offset: StaticIntTuple[rank] = StaticIntTuple[rank](0),
        org_coords_stride: StaticIntTuple[rank] = StaticIntTuple[rank](1),
    ):
        constrained[
            element_layout.all_dims_known(), "Layout must be fully static"
        ]()
        self.ptr = ptr
        self.runtime_layout = runtime_layout
        self.runtime_element_layout = RuntimeLayout[element_layout]()
        self.org_coords_offset = org_coords_offset
        self.org_coords_stride = org_coords_stride

    @always_inline
    fn __init__(
        inout self,
        ptr: UnsafePointer[Scalar[dtype], address_space],
        runtime_layout: RuntimeLayout[layout],
        elemnt_runtime_layout: RuntimeLayout[element_layout],
        /,
        *,
        org_coords_offset: StaticIntTuple[rank] = StaticIntTuple[rank](0),
        org_coords_stride: StaticIntTuple[rank] = StaticIntTuple[rank](1),
    ):
        self.ptr = ptr
        self.runtime_layout = runtime_layout
        self.runtime_element_layout = elemnt_runtime_layout
        self.org_coords_offset = org_coords_offset
        self.org_coords_stride = org_coords_stride

    fn __init__(inout self, *, other: Self):
        """Explicitly copy the provided value.

        Args:
            other: The value to copy.
        """
        self.__copyinit__(other)

    @always_inline
    fn __copyinit__(inout self: Self, existing: Self):
        self.ptr = existing.ptr
        self.runtime_layout = existing.runtime_layout
        self.runtime_element_layout = existing.runtime_element_layout
        self.org_coords_offset = existing.org_coords_offset
        self.org_coords_stride = existing.org_coords_stride

    @always_inline
    fn bitcast[
        new_type: DType,
        /,
        address_space: AddressSpace = Self.address_space,
        element_layout: Layout = Self.element_layout,
    ](self) -> LayoutTensor[
        new_type,
        layout,
        address_space=address_space,
        element_layout=element_layout,
    ]:
        return LayoutTensor[
            new_type,
            layout,
            address_space=address_space,
            element_layout=element_layout,
        ](self.ptr.bitcast[new_type, address_space=address_space]())

    @always_inline
    fn _offset(self, m: Int, n: Int) -> Int:
        return Self.stride[0]() * m + Self.stride[1]() * n

    @always_inline
    fn __getitem__(self, *dims: Int) -> Self.element_type:
        var strides = self.runtime_layout.stride.value
        var vec_res = SIMD[dtype, Self.element_size]()

        # TODO: We should vectorize the reads of contiguous loads this just stash
        # scalar elements.

        @parameter
        for idx in range(Self.element_size):

            @parameter
            if element_layout.all_dims_known():
                alias element_offset = self.element_layout(idx)
                vec_res[idx] = self.ptr.load(
                    Self._getOffset(strides, dims) + element_offset
                )
            else:
                var element_offset = self.runtime_element_layout(idx)
                vec_res[idx] = self.ptr.load(
                    Self._getOffset(strides, dims) + element_offset
                )

        return vec_res

    @always_inline
    fn __setitem__(self, d0: Int, val: Self.element_type):
        var strides = self.runtime_layout.stride.value

        # TODO: We should vectorize contiguous stores this just stash scalars.
        @parameter
        for i in range(Self.element_size):

            @parameter
            if element_layout.all_dims_known():
                alias element_offset = self.element_layout(i)
                self.ptr.store(
                    Self._getOffset(strides, VariadicList[Int](d0))
                    + element_offset,
                    val[i],
                )
            else:
                alias element_offset = self.element_layout(i)
                self.ptr.store(
                    Self._getOffset(strides, VariadicList[Int](d0))
                    + element_offset,
                    val[i],
                )

    @always_inline
    fn __setitem__(self, d0: Int, d1: Int, val: Self.element_type):
        var strides = self.runtime_layout.stride.value

        @parameter
        for i in range(Self.element_size):

            @parameter
            if element_layout.all_dims_known():
                alias element_offset = self.element_layout(i)
                self.ptr.store(
                    Self._getOffset(strides, VariadicList[Int](d0, d1))
                    + element_offset,
                    val[i],
                )
            else:
                var element_offset = self.runtime_element_layout(i)
                self.ptr.store(
                    Self._getOffset(strides, VariadicList[Int](d0, d1))
                    + element_offset,
                    val[i],
                )

    @always_inline
    fn __setitem__(self, d0: Int, d1: Int, d2: Int, val: Self.element_type):
        var strides = self.runtime_layout.stride.value

        @parameter
        for i in range(Self.element_size):

            @parameter
            if element_layout.all_dims_known():
                alias element_offset = self.element_layout(i)
                self.ptr.store(
                    Self._getOffset(strides, VariadicList[Int](d0, d1, d2))
                    + element_offset,
                    val[i],
                )
            else:
                var element_offset = self.runtime_element_layout(i)
                self.ptr.store(
                    Self._getOffset(strides, VariadicList[Int](d0, d1, d2))
                    + element_offset,
                    val[i],
                )

    @always_inline
    fn __setitem__(
        self, d0: Int, d1: Int, d2: Int, d3: Int, val: Self.element_type
    ):
        var strides = self.runtime_layout.stride.value

        @parameter
        for i in range(Self.element_size):

            @parameter
            if element_layout.all_dims_known():
                alias element_offset = self.element_layout(i)
                self.ptr.store(
                    Self._getOffset(strides, VariadicList[Int](d0, d1, d2, d3))
                    + element_offset,
                    val[i],
                )
            else:
                var element_offset = self.runtime_element_layout(i)
                self.ptr.store(
                    Self._getOffset(strides, VariadicList[Int](d0, d1, d2, d3))
                    + element_offset,
                    val[i],
                )

    @always_inline
    fn load[width: Int](self, m: Int, n: Int) -> SIMD[dtype, width]:
        return self.ptr.load[width=width](self._offset(m, n))

    @always_inline
    fn prefetch(self, m: Int, n: Int):
        prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](
            self.ptr.offset(self._offset(m, n))
        )

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
    fn stack_allocation[*, alignment: Int = alignof[dtype]()]() -> Self:
        constrained[layout.all_dims_known(), "Requires fully static layout"]()
        var ptr = stack_allocation[
            layout.size(),
            dtype,
            alignment=alignment,
            address_space=address_space,
        ]()

        return ptr

    @staticmethod
    @always_inline("nodebug")
    fn _toStatic[t: IntTuple]() -> StaticIntTuple[len(t)]:
        var st = StaticIntTuple[len(t)]()

        @parameter
        for i in range(len(t)):
            st[i] = to_int(t[i])
        return st

    @staticmethod
    @always_inline("nodebug")
    fn _getOffset[
        rank: Int
    ](stride: StaticIntTuple[rank], vals: VariadicList[Int]) -> Int:
        var offset = 0

        @parameter
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

        @parameter
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
        dtype,
        __tiled_layout[0],
        address_space=address_space,
    ]:
        alias num_tiles = __get_len[tile_sizes]()

        constrained[
            __tiled_layout[1].rank() == num_tiles,
            "Number of tiles should match the rank",
        ]()

        # Static layout tiling
        # TODO: Consider merge the two cases in away that won't slowdown the fully static layout.
        @parameter
        if __tiled_layout[0].all_dims_known():
            var offset = 0

            @parameter
            for i in range(num_tiles):
                alias stride = to_int(__tiled_layout[1].stride[i])
                offset += tile_coords[i] * stride

            # Update offset to account for tile coords.
            var org_coords_offset = self.org_coords_offset

            @parameter
            for i in range(rank):
                org_coords_offset[i] += tile_sizes[i] * tile_coords[i]

            return LayoutTensor[
                dtype, __tiled_layout[0], address_space=address_space
            ](
                self.ptr.offset(offset),
                org_coords_offset=rebind[
                    StaticIntTuple[__tiled_layout[0].rank()]
                ](org_coords_offset),
            )

        else:
            # Dynamic layout, use strides
            var offset = 0

            var dynamic_layout_shape = RuntimeTuple[__tiled_layout[0].shape]()

            var dynamic_layout_stride = RuntimeTuple[__tiled_layout[0].stride]()

            @parameter
            for i in range(num_tiles):
                var stride = self.runtime_layout.stride.value[i] * tile_sizes[i]
                dynamic_layout_stride.value[
                    i
                ] = self.runtime_layout.stride.value[i]
                offset += tile_coords[i] * stride

            return LayoutTensor[
                dtype,
                __tiled_layout[0],
                address_space=address_space,
            ](
                self.ptr.offset(offset),
                RuntimeLayout(dynamic_layout_shape, dynamic_layout_stride),
            )

    @always_inline
    fn tiled_iterator[
        *tile_sizes: Int,
        axis: Int = 0,
        __tiled_layout: Layout = Self._compute_tile_layout[tile_sizes](),
    ](self, *tile_coords: Int) -> LayoutTensorIter[
        dtype, __tiled_layout[0], address_space, circular=False
    ]:
        alias num_tiles = __get_len[tile_sizes]()

        constrained[
            __tiled_layout[1].rank() == num_tiles,
            "Number of tiles should match the rank",
        ]()

        var ptr_offset = 0

        @parameter
        for i in range(num_tiles):
            alias stride = to_int(__tiled_layout[1].stride[i])
            ptr_offset += tile_coords[i] * stride

        constrained[
            layout.shape[axis].is_value(),
            "The layout in the input axis can't be a tuple",
        ]()

        alias bound = layout.shape[axis].value()
        alias stride = __tiled_layout[1].stride[axis].value()

        return LayoutTensorIter[
            dtype, __tiled_layout[0], address_space, circular=False
        ](self.ptr + ptr_offset, bound, stride=stride)

    @always_inline
    fn split[
        count: Int,
        axis: Int = 0,
        __tile_size: Int = layout.shape[axis].value() // count,
        __tiled_layout: Layout = Self._compute_tile_layout[__tile_size, axis](),
    ](self) -> InlineArray[
        LayoutTensor[
            dtype,
            __tiled_layout[0],
            address_space=address_space,
            element_layout=element_layout,
        ],
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

        var tiles = InlineArray[
            LayoutTensor[
                dtype,
                __tiled_layout[0],
                address_space=address_space,
                element_layout=element_layout,
            ],
            count,
        ](unsafe_uninitialized=True)

        @parameter
        for i in range(count):
            var ptr = UnsafePointer.address_of(tiles.unsafe_get(i))
            ptr.init_pointee_move(
                LayoutTensor[
                    dtype,
                    __tiled_layout[0],
                    address_space=address_space,
                    element_layout=element_layout,
                ](self.ptr.offset(i * __tile_size * stride)),
            )

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
                data_layout, Layout(threads_layout.shape[axis.value()])
            )
        else:
            for dim in threads_layout.shape:
                thread_tile.append(Layout(dim))

            return zipped_divide(data_layout, thread_tile)

    @always_inline
    fn distribute[
        threads_layout: Layout,
        axis: Optional[Int] = None,
        swizzle: Optional[_swizzle_signature] = None,
        tiled_layout: Layout = _compute_distribute_layout[
            layout, threads_layout, axis
        ](),
        submode_axis: Optional[Int] = None,
    ](self, thread_id: UInt) -> LayoutTensor[
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

        alias coalesce_thread_layout = coalesce(threads_layout, keep_rank=True)

        alias res_rank = tiled_layout[1].rank()

        # Update org_coords offset and stride according to thread_id.
        var org_coords_offset = StaticIntTuple[res_rank]()
        var org_coords_stride = StaticIntTuple[res_rank]()

        @parameter
        for i in range(res_rank):
            alias stride_i: UInt = to_int(
                flatten(coalesce_thread_layout.stride)[axis.value()]
            ) if axis else to_int(flatten(coalesce_thread_layout.stride)[i])
            alias shape_i: UInt = to_int(
                flatten(coalesce_thread_layout.shape)[axis.value()]
            ) if axis else to_int(flatten(coalesce_thread_layout.shape)[i])
            var thread_corrrds_i: UInt = (thread_id // stride_i) % shape_i
            org_coords_offset[i] = thread_corrrds_i + self.org_coords_offset[i]
            org_coords_stride[i] = self.org_coords_stride[i] * shape_i

        # Static layout tiling
        # TODO: Consider merge the two cases in away that won't slowdown the fully static layout.
        @parameter
        if layout.all_dims_known():
            alias fragments_layout_stride = flatten(tiled_layout[0].stride)

            alias threads_layout_shape = flatten(threads_layout.shape)
            alias threads_layout_stride = flatten(threads_layout.stride)

            # Only extract coordinates in the given axis.
            # Example: axis = 0 for 2x2 threads, we only need thread 0 and 1's
            # coordinates since thread 2 and 3 are getting the same tile.
            alias thread_projected_stride = flatten(
                threads_layout.stride[
                    axis.value()
                ] if axis else threads_layout.stride
            )
            alias thread_projected_shape = flatten(
                threads_layout.shape[
                    axis.value()
                ] if axis else threads_layout.shape
            )

            var offset: Scalar[Self.index_type] = 0

            @parameter
            for i in range(len(fragments_layout_stride)):
                alias fragments_stride_i: UInt = to_int(
                    fragments_layout_stride[i]
                ).value
                alias shape_i: UInt = to_int(thread_projected_shape[i]).value
                alias stride_i: UInt = to_int(thread_projected_stride[i]).value
                var thread_coord_i: UInt = (thread_id // stride_i) % shape_i
                offset += thread_coord_i * fragments_stride_i

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
            ](
                self.ptr.offset(int(swizzled_offset)),
                org_coords_offset=rebind[
                    StaticIntTuple[tiled_layout[1].rank()]
                ](org_coords_offset),
                org_coords_stride=rebind[
                    StaticIntTuple[tiled_layout[1].rank()]
                ](org_coords_stride),
            )

        else:
            constrained[
                layout.known_shape() and threads_layout.all_dims_known(),
                (
                    "Distribute expecting layout with static shapes and fully"
                    " static threads_layout"
                ),
            ]()
            alias fragments_layout_stride = flatten(tiled_layout[0].stride)

            alias threads_layout_shape = flatten(threads_layout.shape)
            alias threads_layout_stride = flatten(threads_layout.stride)

            # Only extract coordinates in the given axis.
            # Example: axis = 0 for 2x2 threads, we only need thread 0 and 1's
            # coordinates since thread 2 and 3 are getting the same tile.
            alias thread_projected_stride = flatten(
                threads_layout.stride[
                    axis.value()
                ] if axis else threads_layout.stride
            )
            alias thread_projected_shape = flatten(
                threads_layout.shape[
                    axis.value()
                ] if axis else threads_layout.shape
            )

            var offset: Scalar[Self.index_type] = 0

            var runtime_shape = RuntimeTuple[tiled_layout[1].shape]()
            var runtime_stride = RuntimeTuple[tiled_layout[1].stride]()

            @parameter
            for i in range(runtime_shape.scalar_length):
                alias shape_i = to_int(flatten(layout.shape)[i])
                alias thread_shape_i = threads_layout[i].size()
                runtime_stride.value[i] = (
                    self.runtime_layout.stride.value[i] * thread_shape_i
                )

            @parameter
            for i in range(len(flatten(Self.layout.stride))):
                var fragments_stride_i = self.runtime_layout.stride.value[i]
                alias shape_i: UInt = to_int(thread_projected_shape[i]).value
                alias stride_i: UInt = to_int(thread_projected_stride[i]).value
                var thread_coord_i: UInt = (thread_id // stride_i) % shape_i
                offset += thread_coord_i * fragments_stride_i

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
            ](
                self.ptr.offset(int(swizzled_offset)),
                RuntimeLayout(runtime_shape, runtime_stride),
                org_coords_offset=rebind[
                    StaticIntTuple[tiled_layout[1].rank()]
                ](org_coords_offset),
                org_coords_stride=rebind[
                    StaticIntTuple[tiled_layout[1].rank()]
                ](org_coords_stride),
            )

    # Returns the original coordiantes a specific tensor element at `idx`.
    @always_inline
    fn element_coords[idx: Int](self) -> StaticIntTuple[rank]:
        constrained[
            layout.known_shape(),
            "element_coords only support layouts of know shape",
        ]()
        alias layout_coords = Layout(Self.layout.shape)
        alias coords = Self._toStatic[layout_coords.idx2crd(idx)]()
        return (
            self.org_coords_offset
            + rebind[StaticIntTuple[rank]](coords) * self.org_coords_stride
        )

    @always_inline
    fn vectorize[
        *vector_shape: Int,
        __tiled_layout: Layout = Self._compute_tile_layout[vector_shape](),
    ](self) -> LayoutTensor[
        dtype,
        coalesce(__tiled_layout[1], keep_rank=True),
        address_space=address_space,
        element_layout = __tiled_layout[0],
    ]:
        # Update element stride to account for vector shapes.
        var org_coords_stride = StaticIntTuple[rank]()

        @parameter
        for i in range(rank):
            org_coords_stride[i] = vector_shape[i]

        @parameter
        if layout.all_dims_known():
            return LayoutTensor[
                dtype,
                coalesce(__tiled_layout[1], keep_rank=True),
                address_space=address_space,
                element_layout = __tiled_layout[0],
            ](
                self.ptr,
                org_coords_offset=rebind[
                    StaticIntTuple[
                        coalesce(__tiled_layout[1], keep_rank=True).rank()
                    ]
                ](self.org_coords_offset),
                org_coords_stride=rebind[
                    StaticIntTuple[
                        coalesce(__tiled_layout[1], keep_rank=True).rank()
                    ]
                ](org_coords_stride),
            )
        else:
            constrained[
                coalesce(__tiled_layout[0]).known_shape(),
                "Result element layout should have known shape",
            ]()
            var runtime_shape = RuntimeTuple[
                coalesce(__tiled_layout[1], keep_rank=True).shape
            ]()
            var runtime_stride = RuntimeTuple[
                coalesce(__tiled_layout[1], keep_rank=True).stride
            ]()

            var runtime_element_layout_shape = RuntimeTuple[
                __tiled_layout[0].shape
            ]()
            var runtime_element_layout_stride = RuntimeTuple[
                __tiled_layout[0].stride
            ](self.runtime_layout.stride.value)

            @parameter
            for i in range(runtime_shape.scalar_length):
                runtime_shape.value[i] = (
                    self.runtime_layout.shape.value[i] // vector_shape[i]
                )
                runtime_stride.value[i] = (
                    self.runtime_layout.stride.value[i] * vector_shape[i]
                )

            return LayoutTensor[
                dtype,
                coalesce(__tiled_layout[1], keep_rank=True),
                address_space=address_space,
                element_layout = __tiled_layout[0],
            ](
                self.ptr,
                RuntimeLayout(runtime_shape, runtime_stride),
                rebind[RuntimeLayout[__tiled_layout[0]]](
                    RuntimeLayout(
                        runtime_element_layout_shape,
                        runtime_element_layout_stride,
                    )
                ),
                org_coords_offset=rebind[
                    StaticIntTuple[
                        coalesce(__tiled_layout[1], keep_rank=True).rank()
                    ]
                ](self.org_coords_offset),
            )

    @staticmethod
    fn __compute_slice_layout(d0_slice: Slice, d1_slice: Slice) -> Layout:
        constrained[
            layout.shape.__len__() == 2,
            "Only rank-2 tensors slices are supported for now!",
        ]()
        return Layout(
            IntTuple(
                _get_slice_size(Self.layout, d0_slice, 0),
                _get_slice_size(Self.layout, d1_slice, 1),
            ),
            layout.stride,
        )

    @staticmethod
    fn __compute_slice_layout(
        slice_0: Slice, slice_1: Slice, slice_0_axis: Int, slice_1_axis: Int
    ) -> Layout:
        constrained[
            layout.shape.__len__() > 2,
            "Rank should be >= 2",
        ]()
        var sliced_layout = sublayout(Self.layout, slice_0_axis, slice_1_axis)
        return Layout(
            IntTuple(
                _get_slice_size(sliced_layout, slice_0, 0),
                _get_slice_size(sliced_layout, slice_1, 1),
            ),
            sliced_layout.stride,
        )

    @staticmethod
    fn __compute_slice_layout(slice_0: Slice, slice_0_axis: Int) -> Layout:
        constrained[
            layout.shape.__len__() > 1,
            "Rank should be >= 1",
        ]()
        var sliced_layout = sublayout(Self.layout, slice_0_axis)
        return Layout(
            IntTuple(
                _get_slice_size(sliced_layout, slice_0, 0),
            ),
            sliced_layout.stride[0],
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
        alias stride_m = to_int(__slice_layout.stride[0])
        alias stride_n = to_int(__slice_layout.stride[1])

        alias d0_slice_start = d0_slice.start.or_else(0)
        alias d1_slice_start = d1_slice.start.or_else(0)

        var offset = d0_slice_start * stride_m + d1_slice_start * stride_n

        return LayoutTensor[
            dtype,
            __slice_layout,
            address_space=address_space,
            element_layout=element_layout,
        ](self.ptr.offset(offset))

    @always_inline
    fn slice[
        d0_slice: Slice,
        d1_slice: Slice,
        slice_indices: StaticIntTuple[2],
        __offset_dims: Int = Self.rank - 2,
        __slice_layout: Layout = Self.__compute_slice_layout(
            d0_slice, d1_slice, slice_indices[0], slice_indices[1]
        ),
    ](
        self,
        offsets: StaticIntTuple[__offset_dims],
    ) -> LayoutTensor[
        dtype,
        __slice_layout,
        address_space=address_space,
        element_layout=element_layout,
    ]:
        constrained[
            d0_slice.step == 1 and d1_slice.step == 1,
            "Slice should have no gaps",
        ]()
        constrained[
            slice_indices[0] < slice_indices[1],
            "Slice indices should be ordered",
        ]()
        alias stride_0 = to_int(__slice_layout.stride[0])
        alias stride_1 = to_int(__slice_layout.stride[1])

        alias d0_slice_start = d0_slice.start.or_else(0)
        alias d1_slice_start = d1_slice.start.or_else(0)

        var slice_offset = d0_slice_start * stride_0 + d1_slice_start * stride_1

        var idx = 0

        @parameter
        for i in range(Self.rank):
            alias stride_i = to_int(Self.layout.stride[i])

            alias offset_index = _not_in_tuple[i, 2, slice_indices]()

            @parameter
            if offset_index:
                slice_offset += offsets[idx] * stride_i
                idx += 1

        return LayoutTensor[
            dtype,
            __slice_layout,
            address_space=address_space,
            element_layout=element_layout,
        ](self.ptr.offset(slice_offset))

    # FIXME: Can't overload slice, hitting compiler issue.
    # https://linear.app/modularml/issue/MOCO-174
    @always_inline
    fn slice_1d[
        d0_slice: Slice,
        slice_indices: StaticIntTuple[1],
        __offset_dims: Int = Self.rank - 1,
        __slice_layout: Layout = Self.__compute_slice_layout(
            d0_slice, slice_indices[0]
        ),
    ](
        self,
        offsets: StaticIntTuple[__offset_dims],
    ) -> LayoutTensor[
        dtype,
        __slice_layout,
        address_space=address_space,
        element_layout=element_layout,
    ]:
        constrained[
            d0_slice.step == 1,
            "Slice should have no gaps",
        ]()

        alias stride_0 = to_int(__slice_layout.stride[0])

        alias d0_slice_start = d0_slice.start.or_else(0)

        var slice_offset = d0_slice_start * stride_0

        var idx = 0

        @parameter
        for i in range(Self.rank):
            alias stride_i = to_int(Self.layout.stride[i])

            alias offset_index = _not_in_tuple[i, 1, slice_indices]()

            @parameter
            if offset_index:
                slice_offset += offsets[idx] * stride_i
                idx += 1

        return LayoutTensor[
            dtype,
            __slice_layout,
            address_space=address_space,
            element_layout=element_layout,
        ](self.ptr.offset(slice_offset))

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
    ](self) -> LayoutTensor[
        dtype,
        dst_layout,
        address_space=address_space,
        element_layout=element_layout,
    ]:
        return LayoutTensor[
            dtype,
            dst_layout,
            address_space=address_space,
            element_layout=element_layout,
        ](self.ptr)

    @always_inline
    fn composition[
        rhs_layout: Layout,
        dst_layout: Layout = composition(layout, rhs_layout),
    ](self) -> LayoutTensor[
        dtype,
        dst_layout,
        address_space=address_space,
        element_layout=element_layout,
    ]:
        return LayoutTensor[
            dtype,
            dst_layout,
            address_space=address_space,
            element_layout=element_layout,
        ](self.ptr)

    @always_inline
    fn distance(
        self, addr: UnsafePointer[Scalar[dtype], address_space, *_]
    ) -> UInt:
        """Returns the distance from the input address."""

        return UInt(int(self.ptr) - int(addr)) // sizeof[dtype]()

    @always_inline
    fn distance(
        self, src: LayoutTensor[dtype, _, _, address_space=address_space]
    ) -> UInt:
        """Returns the distance from the input address."""

        return UInt(int(self.ptr) - int(src.ptr)) // sizeof[dtype]()

    # Returns the linear index of an elem_i 0 ... size(layout).
    #
    @always_inline
    fn __get_element_idx[elem_i: Int](self) -> Int:
        alias element_size = int(self.element_size)

        @parameter
        if layout.all_dims_known():
            alias idx = make_layout(element_layout, layout)(
                elem_i * element_size
            )
            return idx
        else:
            var idx = make_runtime_layout(
                self.runtime_element_layout, self.runtime_layout
            )(elem_i * element_size)
            return idx

    @always_inline
    fn copy_from[
        dst_coords_bound: Optional[StaticIntTuple[rank]] = None,
        src_coords_bound: Optional[StaticIntTuple[rank]] = None,
    ](self, other: LayoutTensor):
        alias other_layout = other.layout

        alias dst_element_size = int(self.element_size)
        alias src_element_size = int(other.element_size)

        constrained[
            layout.known_shape() and other_layout.known_shape(),
            "copy_from must move data of statically known shape",
        ]()

        alias dst_size = layout.size()
        alias src_size = other_layout.size()
        constrained[
            dst_size == src_size,
            "copy_from should move data of the same size",
        ]()
        constrained[
            dst_element_size == src_element_size, "copy_from should move"
        ]()

        alias has_copy_bounds = dst_coords_bound or src_coords_bound

        @parameter
        @always_inline
        fn __is_in_bound[
            rank: Int
        ](coords: StaticIntTuple[rank], bounds: StaticIntTuple[rank]) -> Bool:
            var in_bound = True

            @parameter
            for dim in range(rank):
                in_bound &= coords[dim] < bounds[dim]
            return in_bound

        @parameter
        @always_inline
        fn __compute_element_bound[
            element_layout: Layout
        ](
            coords: StaticIntTuple[rank], bounds: StaticIntTuple[rank]
        ) -> StaticIntTuple[rank]:
            var element_bound = StaticIntTuple[rank]()

            @parameter
            for dim in range(rank):
                alias dim_size = to_int(element_layout.shape[dim])
                element_bound[dim] = (
                    min(dim_size, bounds[dim] - coords[dim]) if coords[dim]
                    < bounds[dim] else 0
                )
            return element_bound

        @parameter
        @always_inline
        fn __load_element[i: Int]() -> Element[dtype, other.element_layout]:
            var src_idx = other.__get_element_idx[i]()

            @parameter
            if src_element_size != 1 and src_coords_bound.__bool__():
                var element_bounds = __compute_element_bound[
                    other.element_layout
                ](
                    rebind[StaticIntTuple[rank]](other.element_coords[i]()),
                    src_coords_bound.value(),
                )
                return Element[dtype, other.element_layout].masked_load[
                    other.address_space
                ](
                    rebind[UnsafePointer[Scalar[dtype], other.address_space]](
                        other.ptr
                    ).offset(src_idx),
                    element_bounds,
                    other.runtime_element_layout,
                )

            return Element[dtype, other.element_layout].load[
                other.address_space
            ](
                rebind[UnsafePointer[Scalar[dtype], other.address_space]](
                    other.ptr
                ).offset(src_idx),
                other.runtime_element_layout,
            )

        @parameter
        @always_inline
        fn __store_element[
            i: Int
        ](src_element: Element[dtype, other.element_layout]):
            var dst_idx = self.__get_element_idx[i]()

            @parameter
            if dst_element_size != 1 and dst_coords_bound.__bool__():
                var element_bounds = __compute_element_bound[
                    self.element_layout
                ](self.element_coords[i](), dst_coords_bound.value())
                Element[dtype, self.element_layout](
                    rebind[
                        Element[dtype, self.element_layout].element_data_type
                    ](src_element.element_data)
                ).masked_store(self.ptr.offset(dst_idx), element_bounds)
            else:
                Element[dtype, self.element_layout](
                    rebind[
                        Element[dtype, self.element_layout].element_data_type
                    ](src_element.element_data)
                ).store(self.ptr.offset(dst_idx))

        @parameter
        for i in range(dst_size):

            @parameter
            if has_copy_bounds:

                @parameter
                if src_coords_bound.__bool__() and dst_element_size == 1:
                    if not __is_in_bound(
                        self.element_coords[i](), dst_coords_bound.value()
                    ):
                        continue

                @parameter
                if dst_coords_bound.__bool__() and src_element_size == 1:
                    if not __is_in_bound(
                        other.element_coords[i](),
                        rebind[StaticIntTuple[other.rank]](
                            dst_coords_bound.value()
                        ),
                    ):
                        continue

            __store_element[i](__load_element[i]())

    # TODO: Remove when masked tensor is fixed (KERN-607)
    @always_inline
    fn copy_from_masked_src[
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
        offset: Int,
        rows: Int,
        cols: Int,
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

        @parameter
        for i in range(dst_size):
            alias src_idx = make_layout(other.element_layout, other_layout)(
                i * src_element_size
            )
            alias dst_idx = make_layout(self.element_layout, self.layout)(
                i * dst_element_size
            )
            var m: Int
            var n: Int
            m, n = divmod(offset + src_idx, cols)
            if m < rows:
                var src_element = Element[dtype, other.element_layout].load(
                    other.ptr.offset(src_idx)
                )
                alias dst_element_type = Element[dtype, self.element_layout]
                dst_element_type(
                    rebind[dst_element_type.element_data_type](
                        src_element.element_data
                    )
                ).store(self.ptr.offset(dst_idx))

    # TODO: Remove when masked tensor is fixed (KERN-607)
    @always_inline
    fn copy_from_masked_dst[
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
        offset: Int,
        rows: Int,
        cols: Int,
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

        @parameter
        for i in range(dst_size):
            alias src_idx = make_layout(other.element_layout, other_layout)(
                i * src_element_size
            )
            alias dst_idx = make_layout(self.element_layout, self.layout)(
                i * dst_element_size
            )
            var m: Int
            var n: Int
            m, n = divmod(offset + src_idx, cols)
            if m < rows:
                var src_element = Element[dtype, other.element_layout].load(
                    other.ptr.offset(src_idx)
                )
                alias dst_element_type = Element[dtype, self.element_layout]
                dst_element_type(
                    rebind[dst_element_type.element_data_type](
                        src_element.element_data
                    )
                ).store(self.ptr.offset(dst_idx))

    @always_inline
    fn copy_from(
        self,
        other: LayoutTensor,
        offset: Int,
        rows: Int,
        cols: Int,
    ):
        alias other_layout = other.layout

        alias dst_size = layout.size()
        alias src_size = other_layout.size()

        alias dst_element_size = int(self.element_size)
        alias src_element_size = int(other.element_size)

        constrained[
            layout.known_shape() and other_layout.known_shape(),
            "copy_from must move data of statically known shape",
        ]()

        constrained[
            dst_size == src_size, "copy_from should move data of the same size"
        ]()

        constrained[
            dst_element_size == src_element_size, "copy_from should move"
        ]()

        @parameter
        for i in range(dst_size):
            alias src_idx = make_layout(other.element_layout, other_layout)(
                i * src_element_size
            )
            alias dst_idx = make_layout(self.element_layout, self.layout)(
                i * dst_element_size
            )
            var m: Int
            var n: Int
            m, n = divmod(offset + dst_idx, cols)
            if m < rows:
                var src_element = Element[dtype, other.element_layout].load[
                    other.address_space
                ](
                    rebind[UnsafePointer[Scalar[dtype], other.address_space]](
                        other.ptr
                    ).offset(src_idx),
                    other.runtime_element_layout,
                )
                alias dst_element_type = Element[dtype, self.element_layout]
                dst_element_type(
                    rebind[dst_element_type.element_data_type](
                        src_element.element_data
                    )
                ).store(self.ptr.offset(dst_idx))

    @always_inline
    fn copy_from_async[
        src_layout: Layout,
        src_addr_space: AddressSpace,
        src_element_layout: Layout,
        masked: Bool = False,
    ](
        self,
        src: LayoutTensor[
            dtype,
            src_layout,
            address_space=src_addr_space,
            element_layout=src_element_layout,
        ],
        src_idx_bound: Int = UNKNOWN_VALUE,
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

        # Coalesce element layouts to simplify vectorization condition.
        alias coalesce_src_element_layout = coalesce(src_element_layout)
        alias coalesce_dst_element_layout = coalesce(self.element_layout)

        @parameter
        if (
            coalesce_src_element_layout.rank() == 1
            and coalesce_dst_element_layout.stride[0] == 1
            and coalesce_dst_element_layout.rank() == 1
            and coalesce_dst_element_layout.stride[0] == 1
        ):

            @parameter
            for i in range(dst_size):
                alias src_idx = make_layout(src.element_layout, src_layout)(
                    i * src_element_size
                )
                alias dst_idx = make_layout(self.element_layout, self.layout)(
                    i * dst_element_size
                )

                @parameter
                if masked:
                    var src_copy_size = element_size_bytes if src_idx < src_idx_bound else 0
                    async_copy[element_size_bytes](
                        src_ptr + src_idx, dst_ptr + dst_idx, src_copy_size
                    )
                else:
                    async_copy[element_size_bytes](
                        src_ptr + src_idx, dst_ptr + dst_idx
                    )

        else:

            @parameter
            for i in range(dst_size * dst_element_size):
                alias src_idx = make_layout(src.element_layout, src_layout)(i)
                alias dst_idx = make_layout(self.element_layout, self.layout)(i)

                async_copy[4](src_ptr + src_idx, dst_ptr + dst_idx)

    @always_inline
    fn copy_from_async_masked_src[
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
        offset: Int,
        rows: Int,
        cols: Int,
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

        # Coalesce element layouts to simplify vectorization condition.
        alias coalesce_src_element_layout = coalesce(src_element_layout)
        alias coalesce_dst_element_layout = coalesce(self.element_layout)

        @parameter
        if (
            coalesce_src_element_layout.rank() == 1
            and coalesce_dst_element_layout.stride[0] == 1
            and coalesce_dst_element_layout.rank() == 1
            and coalesce_dst_element_layout.stride[0] == 1
        ):

            @parameter
            for i in range(dst_size):
                alias src_idx = make_layout(src.element_layout, src_layout)(
                    i * src_element_size
                )
                alias dst_idx = make_layout(self.element_layout, self.layout)(
                    i * dst_element_size
                )

                var m: Int
                var n: Int
                m, n = divmod(offset + src_idx, cols)
                if m < rows:
                    async_copy[element_size_bytes](
                        src_ptr + src_idx, dst_ptr + dst_idx
                    )
        else:

            @parameter
            for i in range(dst_size * dst_element_size):
                alias src_idx = make_layout(src.element_layout, src_layout)(i)
                alias dst_idx = make_layout(self.element_layout, self.layout)(i)

                var m: Int
                var n: Int
                m, n = divmod(offset + src_idx, cols)
                if m < rows:
                    async_copy[4](src_ptr + src_idx, dst_ptr + dst_idx)

    fn linspace(self):
        @parameter
        if len(layout) == 1:
            for m in range(self.runtime_layout.shape[0].value[0]):
                self.ptr[m] = m

        elif len(layout) == 2:
            for m in range(self.runtime_layout.shape[0].value[0]):
                for n in range(self.runtime_layout.shape[1].value[0]):
                    self[m, n] = m * self.runtime_layout.shape[1].value[0] + n
        else:
            abort("LayoutTensor linspace only support rank 1-2 layouts.")

    @always_inline
    fn fill(self, val: Scalar[dtype]):
        @parameter
        if layout.all_dims_known():
            alias num_elements = layout.size() * Self.element_size

            @parameter
            for i in range(num_elements):
                self.ptr[i] = val
        else:
            var num_elements = self.runtime_layout.size() * Self.element_size
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
            for m in range(self.runtime_layout.shape[0].value[0]):
                for n in range(self.runtime_layout.shape[1].value[0]):
                    print(self[m, n], end=" ")
                print("")
        else:
            for i in range(layout.size()):
                var vec_offset = self.runtime_layout(i)
                var vec = SIMD[dtype, Self.element_size]()

                @parameter
                for idx in range(Self.element_size):
                    alias element_offset = self.element_layout(idx)
                    vec[idx] = self.ptr.load(vec_offset + element_offset)

                print(vec)


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

    @parameter
    for i in range(M):

        @parameter
        for j in range(N):
            res[i, j] += lhs[i].cast[dtype]() * rhs[j].cast[dtype]()


# Synchronous copy from DRAM -> SRAM, this requires w/r thread affinity mapping.
#
@always_inline
fn copy_dram_to_sram[
    src_layout: Layout,
    dst_layout: Layout,
    dtype: DType,
    src_thread_layout: Layout,
    dst_thread_layout: Layout,
    src_element_layout: Layout,
    dst_element_layout: Layout,
    swizzle: Optional[_swizzle_signature] = None,
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
    var src_fragments = src.distribute[src_thread_layout](ThreadIdx.x())
    var dst_fragments = dst.distribute[dst_thread_layout, swizzle=swizzle](
        ThreadIdx.x()
    )
    dst_fragments.copy_from(src_fragments)


# Synchronous copy from DRAM -> SRAM, this requires w/r thread affinity mapping.
#
@always_inline
fn copy_dram_to_sram[
    src_layout: Layout,
    dst_layout: Layout,
    dtype: DType,
    thread_layout: Layout,
    src_element_layout: Layout,
    dst_element_layout: Layout,
    swizzle: Optional[_swizzle_signature] = None,
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
    copy_dram_to_sram[
        src_layout,
        dst_layout,
        dtype,
        thread_layout,
        thread_layout,
        src_element_layout,
        dst_element_layout,
        swizzle,
    ](dst, src)


# Asynchronous copy from DRAM -> SRAM, this requires w/r thread affinity mapping.
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
    swizzle: Optional[_swizzle_signature] = None,
    masked: Bool = False,
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
    num_rows: Int = UNKNOWN_VALUE,
):
    var src_fragments = src.distribute[src_thread_layout](ThreadIdx.x())
    var dst_fragments = dst.distribute[dst_thread_layout, swizzle=swizzle](
        ThreadIdx.x()
    )

    @parameter
    if not masked:
        dst_fragments.copy_from_async(src_fragments)
    else:
        constrained[
            src_layout.stride[1].value() == src.element_size
            and src_layout.rank() == 2,
            "Only support masking rows and 2D row major layout.",
        ]()
        var src_frag_offset = src_fragments.distance(src.ptr)
        alias stride = src_layout.stride[0].value()
        var src_idx_bound = num_rows * stride - src_frag_offset
        dst_fragments.copy_from_async[masked=True](src_fragments, src_idx_bound)


# Asynchronous copy from DRAM -> SRAM, this requires w/r thread affinity mapping.
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
    swizzle: Optional[_swizzle_signature] = None,
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
    offset: Int,
    rows: Int,
    cols: Int,
):
    var src_framgents = src.distribute[src_thread_layout](ThreadIdx.x())
    var dst_framgents = dst.distribute[dst_thread_layout, swizzle=swizzle](
        ThreadIdx.x()
    )
    var thrd_offset = offset + src.distribute[src_thread_layout](
        ThreadIdx.x()
    ).distance(src.ptr)
    dst_framgents.copy_from_async_masked_src(
        src_framgents, thrd_offset, rows, cols
    )


# Asynchronous copy from DRAM -> SRAM, this requires w/r thread affinity mapping.
#
@always_inline
fn copy_dram_to_sram_async[
    src_layout: Layout,
    dst_layout: Layout,
    dtype: DType,
    thread_layout: Layout,
    src_element_layout: Layout,
    dst_element_layout: Layout,
    swizzle: Optional[_swizzle_signature] = None,
    masked: Bool = False,
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
    num_rows: Int = UNKNOWN_VALUE,
):
    copy_dram_to_sram_async[
        src_layout,
        dst_layout,
        dtype,
        thread_layout,
        thread_layout,
        src_element_layout,
        dst_element_layout,
        swizzle,
        masked,
    ](dst, src, num_rows)


# Asynchronous copy from DRAM -> SRAM, this requires w/r thread affinity mapping.
#
@always_inline
fn copy_dram_to_sram_async[
    src_layout: Layout,
    dst_layout: Layout,
    dtype: DType,
    thread_layout: Layout,
    src_element_layout: Layout,
    dst_element_layout: Layout,
    swizzle: Optional[_swizzle_signature] = None,
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
    offset: Int,
    rows: Int,
    cols: Int,
):
    copy_dram_to_sram_async[
        src_layout,
        dst_layout,
        dtype,
        thread_layout,
        thread_layout,
        src_element_layout,
        dst_element_layout,
        swizzle,
    ](dst, src, offset, rows, cols)


@always_inline
fn copy_sram_to_dram[
    src_layout: Layout,
    dst_layout: Layout,
    src_type: DType,
    dst_type: DType,
    thread_layout: Layout,
    src_element_layout: Layout,
    dst_element_layout: Layout,
    swizzle: Optional[_swizzle_signature] = None,
](
    dst: LayoutTensor[
        dst_type,
        dst_layout,
        address_space = _GPUAddressSpace.GENERIC,
        element_layout=dst_element_layout,
    ],
    src: LayoutTensor[
        src_type,
        src_layout,
        address_space = _GPUAddressSpace.SHARED,
        element_layout=src_element_layout,
    ],
):
    var src_fragments = src.distribute[thread_layout](ThreadIdx.x())
    var dst_fragments = dst.distribute[thread_layout, swizzle=swizzle](
        ThreadIdx.x()
    )

    @parameter
    if src_type == dst_type:
        dst_fragments.copy_from(src_fragments.bitcast[dst_type]())
    else:
        constrained[
            src_type == DType.float32 and dst_type.is_half_float(),
            "Only support FP32 -> half precision downcast during copy.",
        ]()

        alias simd_size = simdwidthof[dst_type]()
        # TODO: generalize the copy to non-scalar case if possible.
        constrained[
            src_element_layout.size() == simd_size
            and dst_element_layout.size() == simd_size,
            "Only FP32 -> half precision downcast for vectorized copy.",
        ]()

        alias num_stores_per_thread = dst_fragments.layout.size()
        alias src_align = alignof[SIMD[src_type, simdwidthof[src_type]()]]()
        alias dst_align = alignof[SIMD[dst_type, simd_size]]()

        @parameter
        for i in range(num_stores_per_thread):
            alias src_idx = src_fragments.layout(i)
            alias dst_idx = dst_fragments.layout(i)

            var src_vec = (src_fragments.ptr + src_idx).load[
                width=simd_size, alignment=src_align
            ]()
            (dst_fragments.ptr + dst_idx).store[alignment=dst_align](
                src_vec.cast[dst_type]()
            )


@always_inline
fn copy_sram_to_dram[
    src_layout: Layout,
    dst_layout: Layout,
    src_type: DType,
    dst_type: DType,
    thread_layout: Layout,
    src_element_layout: Layout,
    dst_element_layout: Layout,
    swizzle: Optional[_swizzle_signature] = None,
](
    dst: LayoutTensor[
        dst_type,
        dst_layout,
        address_space = _GPUAddressSpace.GENERIC,
        element_layout=dst_element_layout,
    ],
    src: LayoutTensor[
        src_type,
        src_layout,
        address_space = _GPUAddressSpace.SHARED,
        element_layout=src_element_layout,
    ],
    offset: Int,
    rows: Int,
    cols: Int,
):
    var src_fragments = src.distribute[thread_layout](ThreadIdx.x())
    var dst_fragments = dst.distribute[thread_layout, swizzle=swizzle](
        ThreadIdx.x()
    )
    var thread_offset = offset + dst_fragments.distance(dst.ptr)

    @parameter
    if src_type == dst_type:
        dst_fragments.copy_from(src_fragments.bitcast[dst_type]())
    else:
        constrained[
            src_type == DType.float32 and dst_type.is_half_float(),
            "Only support FP32 -> half precision downcast during copy.",
        ]()

        alias simd_size = simdwidthof[dst_type]()
        # TODO: generalize the copy to non-scalar case if possible.
        constrained[
            src_element_layout.size() == simd_size
            and dst_element_layout.size() == simd_size,
            "Only FP32 -> half precision downcast for vectorized copy.",
        ]()

        alias num_stores_per_thread = dst_fragments.layout.size()
        alias src_align = alignof[SIMD[src_type, simdwidthof[src_type]()]]()
        alias dst_align = alignof[SIMD[dst_type, simd_size]]()

        @parameter
        for i in range(num_stores_per_thread):
            alias src_idx = src_fragments.layout(i)
            alias dst_idx = dst_fragments.layout(i)

            var m: Int
            var n: Int
            m, n = divmod(thread_offset + src_idx, cols)
            if m < rows:
                var src_vec = (src_fragments.ptr + src_idx).load[
                    width=simd_size, alignment=src_align
                ]()
                dst_fragments.ptr.store[alignment=dst_align](
                    dst_idx, src_vec.cast[dst_type]()
                )


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
            src_warp_layout, axis = axis.value()
        ](ThreadIdx.x())
        dst.copy_from(src_fragments)
    else:
        var src_fragments = src.distribute[src_warp_layout](ThreadIdx.x())
        dst.copy_from(src_fragments)


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
    src_addr_space: AddressSpace,
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
        address_space=src_addr_space,
        element_layout=src_element_layout,
    ],
):
    var dst_fragments = dst.distribute[dst_thread_layout](ThreadIdx.x())
    dst_fragments.copy_from(src)


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
    offset: Int,
    rows: Int,
    cols: Int,
):
    var dst_framgents = dst.distribute[dst_thread_layout](ThreadIdx.x())
    var thrd_offset = dst.distribute[dst_thread_layout](ThreadIdx.x()).distance(
        dst.ptr
    ) + offset
    dst_framgents.copy_from_masked_dst(src, thrd_offset, rows, cols)


@always_inline
fn copy_local_to_sram[
    src_layout: Layout,
    dst_layout: Layout,
    src_type: DType,
    dst_type: DType,
    thread_layout: Layout,
    src_element_layout: Layout,
    dst_element_layout: Layout,
    src_addr_space: AddressSpace,
](
    dst: LayoutTensor[
        dst_type,
        dst_layout,
        address_space = _GPUAddressSpace.SHARED,
        element_layout=dst_element_layout,
    ],
    src: LayoutTensor[
        src_type,
        src_layout,
        address_space=src_addr_space,
        element_layout=src_element_layout,
    ],
):
    var dst_frag = dst.distribute[thread_layout](ThreadIdx.x())

    @parameter
    if src_type == dst_type:
        dst_frag.copy_from(src)
    else:
        constrained[
            src_type == DType.float32 and dst_type.is_half_float(),
            "Only support FP32 -> half precision downcast during copy.",
        ]()

        constrained[
            src.element_size == dst.element_size,
            "src and dst element size mismatch.",
        ]()

        alias num_stores_per_thread = dst_frag.layout.size()
        alias elem_size = src.element_size

        @parameter
        for i in range(num_stores_per_thread):
            var src_vec = src.aligned_load[elem_size](i, 0)
            var dst_vec: SIMD[dst_type, elem_size] = 0
            alias dst_idx = dst_frag.layout(i)

            @parameter
            for j in range(0, elem_size, 2):
                var vec_converted = SIMD[src_type, 2](
                    src_vec[j], src_vec[j + 1]
                ).cast[dst_type]()
                dst_vec[j] = vec_converted[0]
                dst_vec[j + 1] = vec_converted[1]

            (dst_frag.ptr + dst_idx).store[
                width=elem_size,
                alignment = alignof[SIMD[dst_type, src.element_size]](),
            ](dst_vec)


@always_inline
fn copy_local_to_local[
    dst_type: DType,
    src_type: DType,
    dst_layout: Layout,
    src_layout: Layout,
    dst_element_layout: Layout,
    src_addr_space: AddressSpace,
](
    dst: LayoutTensor[
        dst_type,
        dst_layout,
        address_space = _GPUAddressSpace.LOCAL,
        element_layout=dst_element_layout,
    ],
    src: LayoutTensor[src_type, src_layout, address_space=src_addr_space],
):
    constrained[
        dst.dtype.is_half_float() and src.dtype == DType.float32,
        "Only support copy float32 to bfloat16 for now",
    ]()

    constrained[
        dst.layout.size() == src.layout.size(),
        "dst and src should have the same size.",
    ]()

    # Fast for 2D fragments
    @parameter
    if (
        dst.rank == 2
        and src.rank == 2
        and dst.stride[1]() == 1
        and src.stride[1]() == 1
    ):
        # This path is to map 16x8x16 mma output (16x8) to 16x8x16 mma input (16x16).
        # Output fragment has layout [2 * num_m_mmas, 4]
        # Input  fragment has layout [num_m_mmas, 8]
        alias num_mmas = src_layout.shape[0].value()
        alias src_frag_size = src_layout.shape[1].value()
        alias a_frag_layout = composition(
            src_layout,
            make_layout(Layout.row_major(num_mmas // 2, 2), src_layout[1]),
        )
        # [num_m_mmas, 8] vectorized and transposed to [2, num_m_mmas] x 4
        var dst_vectorized = dst.vectorize[1, src_frag_size]().transpose()
        # [2*num_m_mmas, 4] reshaped and vectorized row_major(num_m_mmas, 2) x 4
        var src_vectorized = src.reshape[a_frag_layout]().vectorize[
            1, src_frag_size
        ]()

        @parameter
        for i in range(dst_vectorized.layout.size()):
            alias dst_idx = dst_vectorized.layout(i)
            alias src_idx = src_vectorized.layout(i)

            var src_vec = (src_vectorized.ptr + src_idx).load[
                width=src_frag_size
            ]()
            var dst_vec = SIMD[dst_type, src_frag_size](0.0)

            @parameter
            for j in range(0, src_frag_size, 2):
                var vec_converted = SIMD[src_type, 2](
                    src_vec[j], src_vec[j + 1]
                ).cast[dst_type]()
                dst_vec[j] = vec_converted[0]
                dst_vec[j + 1] = vec_converted[1]

            (dst_vectorized.ptr + dst_idx).store[width=src_frag_size](dst_vec)

    # Default elementwise copy
    else:

        @parameter
        for i in range(dst.layout.size()):
            alias dst_idx = dst.layout(i)
            alias src_idx = src.layout(i)
            var src_val = src.ptr[src_idx]
            (dst.ptr + dst_idx).store(src_val.cast[dst.dtype]())


# ===-----------------------------------------------------------------------===#
# LayoutTensorIter                                                             #
# ===-----------------------------------------------------------------------===#


@register_passable
struct LayoutTensorIter[
    type: DType,
    layout: Layout,
    address_space: AddressSpace = AddressSpace.GENERIC,
    circular: Bool = False,
]:
    """Iterate through a memory buffer and construct layout tensor.

    The returned layout tensor is NOT vectorized. User should explicitly vectorize.

    TODO: support constructing iterator from layout tensor.
    """

    var ptr: UnsafePointer[Scalar[type], address_space]
    var offset: Int
    var stride: Int
    var bound: Int

    @always_inline
    fn __init__(
        inout self,
        ptr: UnsafePointer[Scalar[type], address_space],
        bound: Int,
        stride: Int = layout.size(),
        offset: Int = 0,
    ):
        self.ptr = ptr
        self.offset = offset
        self.stride = stride
        self.bound = bound

    @always_inline
    fn __copyinit__(inout self: Self, existing: Self):
        self.ptr = existing.ptr
        self.offset = existing.offset
        self.stride = existing.stride
        self.bound = existing.bound

    @always_inline
    fn get(self) -> LayoutTensor[type, layout, address_space=address_space]:
        """Return the layout tensor at current iterator."""
        # TODO: Use deref `[]` to be consistent with mojo feature.

        return LayoutTensor[type, layout, address_space=address_space](
            self.ptr + self.offset
        )

    @always_inline
    fn __iadd__[T: Intable](inout self, rhs: T):
        """Increment the iterator.

        This function is unsafe. It omits bound checking for performance reasons.
        Caller must make sure index doesn't go out-of-bound.
        """

        self.offset += int(rhs) * self.stride

        @parameter
        if circular:
            self.offset = self.offset % self.bound

    @always_inline
    fn __iadd__(inout self, rhs: UInt):
        """Increment the iterator.

        This function is unsafe. It omits bound checking for performance reasons.
        Caller must make sure index doesn't go out-of-bound.
        """
        self += int(rhs)

    @always_inline
    fn next[T: Intable](self, rhs: T) -> Self:
        """Return an iterator pointing to the next `rhs` layout tensor."""

        var next_offset = self.offset + int(rhs) * self.stride

        @parameter
        if circular:
            next_offset = next_offset % self.bound

        return LayoutTensorIter[
            type, layout, address_space=address_space, circular=circular
        ](self.ptr, self.bound, stride=self.stride, offset=next_offset)

    @always_inline
    fn next(self, rhs: UInt = 1) -> Self:
        return self.next(int(rhs))
