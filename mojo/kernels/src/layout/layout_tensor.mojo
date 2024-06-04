# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections.optional import Optional, OptionalReg
from sys.info import sizeof
from sys.intrinsics import PrefetchOptions
from utils import InlineArray, StaticIntTuple
from utils.numerics import max_finite

from algorithm import vectorize
from gpu.memory import async_copy
from gpu.id import ThreadIdx
from memory import memcpy
from memory.unsafe import DTypePointer
from memory.reference import AddressSpace, _GPUAddressSpace

from .int_tuple import flatten, idx2crd, to_int, product, fill_like
from .layout import *
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


# Returns the size of variadic integer parameters.
#
fn __get_len[*var_int: Int]() -> Int:
    return __mlir_op.`pop.variadic.size`(var_int)


# Returns True if shape isn't an integer multiple of tile_sizes, otherwise
# returns False.
#
fn _need_mask[*tile_sizes: Int](shape: IntTuple) -> Bool:
    var no_mask = True

    @parameter
    for i in range(__get_len[tile_sizes]()):
        alias tile_size = tile_sizes[i]
        no_mask = no_mask and (to_int(shape[i]) % tile_size == 0)

    return not no_mask


# Returns the size of the slice in layout dim.
#
fn _get_slice_size(layout: Layout, slice: Slice, dim: Int) -> Int:
    var end = slice.end if slice._has_end() else to_int(layout.shape[dim])
    return end - slice.start


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
    masked: Bool = False,
](CollectionElement, CollectionElementNew):
    var ptr: DTypePointer[dtype, address_space]

    # When LayoutTensor is masked, we need to store three quantities:
    # To specify the per dim original coordinates bounds.
    var max_dim: StaticIntTuple[rank]
    # To specify the per dim offset for the current tile.
    var dim_offset: StaticIntTuple[rank]
    # To specify the per dim stride for the current tile.
    var dim_stride: StaticIntTuple[rank]

    alias element_size = element_layout.size()
    alias element_type = SIMD[dtype, Self.element_size]

    # ===------------------------------------------------------------------=== #
    # Life cycle methods
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __init__(
        inout self,
        ptr: DTypePointer[dtype, address_space],
    ):
        self.ptr = ptr

        self.max_dim = StaticIntTuple[rank](Int.MAX)
        self.dim_offset = StaticIntTuple[rank](0)
        self.dim_stride = StaticIntTuple[rank](1)

    fn __init__(inout self, *, other: Self):
        """Explicitly copy the provided value.

        Args:
            other: The value to copy.
        """
        self.__copyinit__(other)

    @always_inline
    fn __copyinit__(inout self: Self, existing: Self):
        self.ptr = existing.ptr
        self.max_dim = existing.max_dim
        self.dim_offset = existing.dim_offset
        self.dim_stride = existing.dim_stride

    @always_inline
    fn _offset(self, m: Int, n: Int) -> Int:
        return Self.stride[0]() * m + Self.stride[1]() * n

    # Returns True if the idx is accessable (not masked), otherwise False.
    #
    @always_inline
    fn _is_not_masked_elemenet[idx: IntTuple](self) -> Bool:
        var can_access = True
        alias rank = len(idx)

        @parameter
        for i in range(rank):
            alias dim = to_int(idx[i])
            var tile_offset_i = self.dim_offset[i]
            var tile_stride_i = self.dim_stride[i]
            var offset = tile_offset_i + dim * tile_stride_i
            can_access = can_access and offset < self.max_dim[i]

        return can_access

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
        for idx in range(Self.element_size):
            alias element_offset = self.element_layout(idx)
            vec_res[idx] = Scalar.load(
                self.ptr, Self._getOffset(strides, dims) + element_offset
            )

        return vec_res

    @always_inline
    fn __setitem__(self, d0: Int, val: Self.element_type):
        alias strides = Self._toStatic[flatten(layout.stride)]()

        # TODO: We should vectorize contiguous stores this just stash scalars.
        @parameter
        for i in range(Self.element_size):
            alias element_offset = self.element_layout(i)
            SIMD.store(
                self.ptr,
                Self._getOffset(strides, VariadicList[Int](d0))
                + element_offset,
                val[i],
            )

    @always_inline
    fn __setitem__(self, d0: Int, d1: Int, val: Self.element_type):
        alias strides = Self._toStatic[flatten(layout.stride)]()

        @parameter
        for i in range(Self.element_size):
            alias element_offset = self.element_layout(i)
            SIMD.store(
                self.ptr,
                Self._getOffset(strides, VariadicList[Int](d0, d1))
                + element_offset,
                val[i],
            )

    @always_inline
    fn __setitem__(self, d0: Int, d1: Int, d2: Int, val: Self.element_type):
        alias strides = Self._toStatic[flatten(layout.stride)]()

        @parameter
        for i in range(Self.element_size):
            alias element_offset = self.element_layout(i)
            SIMD.store(
                self.ptr,
                Self._getOffset(strides, VariadicList[Int](d0, d1, d2))
                + element_offset,
                val[i],
            )

    @always_inline
    fn __setitem__(
        self, d0: Int, d1: Int, d2: Int, d3: Int, val: Self.element_type
    ):
        alias strides = Self._toStatic[flatten(layout.stride)]()

        @parameter
        for i in range(Self.element_size):
            alias element_offset = self.element_layout(i)
            SIMD.store(
                self.ptr,
                Self._getOffset(strides, VariadicList[Int](d0, d1, d2, d3))
                + element_offset,
                val[i],
            )

    @always_inline
    fn load[width: Int](self, m: Int, n: Int) -> SIMD[dtype, width]:
        return SIMD[size=width].load(self.ptr, self._offset(m, n))

    @always_inline
    fn prefetch(self, m: Int, n: Int):
        SIMD.prefetch[
            PrefetchOptions().for_read().high_locality().to_data_cache()
        ](self.ptr.offset(self._offset(m, n)))

    @always_inline
    fn aligned_load[width: Int](self, m: Int, n: Int) -> SIMD[dtype, width]:
        alias alignment = alignof[SIMD[dtype, width]]()
        return SIMD[size=width].load[alignment=alignment](
            self.ptr, self._offset(m, n)
        )

    @always_inline
    fn store[width: Int](self, m: Int, n: Int, val: SIMD[dtype, width]):
        return SIMD[size=width].store(self.ptr, self._offset(m, n), val)

    @always_inline
    fn aligned_store[width: Int](self, m: Int, n: Int, val: SIMD[dtype, width]):
        alias alignment = alignof[SIMD[dtype, width]]()
        return SIMD[size=width].store[alignment=alignment](
            self.ptr, self._offset(m, n), val
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
        __need_mask: Bool = _need_mask[tile_sizes](Self.layout.shape),
    ](self, *tile_coords: Int) -> LayoutTensor[
        dtype,
        __tiled_layout[0],
        address_space=address_space,
        masked=__need_mask,
    ]:
        alias num_tiles = __get_len[tile_sizes]()

        constrained[
            __tiled_layout[1].rank() == num_tiles,
            "Number of tiles should match the rank",
        ]()

        var offset = 0

        @parameter
        for i in range(num_tiles):
            alias stride = to_int(__tiled_layout[1].stride[i])
            offset += tile_coords[i] * stride

        var res = LayoutTensor[
            dtype,
            __tiled_layout[0],
            address_space=address_space,
            masked=__need_mask,
        ](self.ptr.offset(offset))

        # If not masked and tiles are not an integer multiple of the shape,
        # update the bounds to the pre-tiling shape.
        @parameter
        if not Self.masked and __need_mask:

            @parameter
            for i in range(Self.rank):
                res.max_dim[i] = to_int(Self.layout.shape[i])
                alias tile_size_i = tile_sizes[i]
                res.dim_offset[i] = tile_size_i * tile_coords[i]

        return res

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
            UnsafePointer.address_of(
                tiles._get_reference_unsafe(i)[]
            ).init_pointee_move(
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
    ](self, thread_id: Int) -> LayoutTensor[
        dtype,
        tiled_layout[1],
        address_space=address_space,
        element_layout=element_layout,
        masked = Self.masked,
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

        # FIXME: We should set everything once, but its better to fill the
        # masking data inplace here.
        var res = LayoutTensor[
            dtype,
            tiled_layout[1],
            address_space=address_space,
            element_layout=element_layout,
            masked = Self.masked,
        ](self.ptr.offset(offset))

        @parameter
        for i in range(len(fragments_layout_stride)):
            alias fragments_stride_i = to_int(fragments_layout_stride[i])
            alias shape_i = to_int(thread_projected_shape[i])
            alias stride_i = to_int(thread_projected_stride[i])
            var thread_coord_i = (thread_id // stride_i) % shape_i
            offset += thread_coord_i * fragments_stride_i

            # Populate data needed for masked access.
            @parameter
            if Self.masked:
                res.max_dim[i] = self.max_dim[i]
                res.dim_offset[i] = self.dim_offset[i] + thread_coord_i
                res.dim_stride[i] = shape_i

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

        # Adjust to actual offset.
        res.ptr = res.ptr.offset(swizzled_offset)
        return res

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
        var offset = d0_slice.start * stride_m + d1_slice.start * stride_n
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

        var slice_offset = d0_slice.start * stride_0 + d1_slice.start * stride_1

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

        var slice_offset = d0_slice.start * stride_0

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
    fn distance(self, addr: DTypePointer[dtype, address_space]) -> Int:
        """Returns the distance from the input address."""

        return (int(self.ptr) - int(addr)) // sizeof[dtype]()

    @always_inline
    fn distance(
        self, src: LayoutTensor[dtype, _, _, address_space=address_space]
    ) -> Int:
        """Returns the distance from the input address."""

        return (int(self.ptr) - int(src.ptr)) // sizeof[dtype]()

    @always_inline
    fn copy_from[
        other_layout: Layout,
        other_addr_space: AddressSpace,
        other_element_layout: Layout,
        other_mask: Bool,
    ](
        self,
        other: LayoutTensor[
            dtype,
            other_layout,
            address_space=other_addr_space,
            element_layout=other_element_layout,
            masked=other_mask,
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

        constrained[
            not other_mask or dst_element_size == 1,
            "For masked src only scalar copy is supported",
        ]()

        # Vectorize 1-D element read/writes.
        @parameter
        if (
            other_element_layout.rank() == 1
            and other_element_layout.stride[0] == 1
            and self.element_layout.rank() == 1
            and self.element_layout.stride[0] == 1
            and dst_element_size > 1  #
        ):

            @parameter
            for i in range(dst_size):
                alias src_idx = make_layout(other.element_layout, other_layout)(
                    i * src_element_size
                )
                alias dst_idx = make_layout(self.element_layout, self.layout)(
                    i * dst_element_size
                )
                var src_vec = rebind[self.element_type](
                    SIMD[size=src_element_size].load[
                        alignment = alignof[other.element_type]()
                    ](other.ptr, src_idx)
                )
                SIMD[size = self.element_size].store[
                    alignment = alignof[self.element_type](),
                ](self.ptr, dst_idx, src_vec)

        # Vector read scalar writes.
        elif (
            other_element_layout.rank() == 1
            and other_element_layout.stride[0] == 1
            and self.element_layout.rank() == 1
            and self.element_layout.stride[0] != 1
        ):

            @parameter
            for i in range(src_size):
                alias src_idx = make_layout(other.element_layout, other_layout)(
                    i * src_element_size
                )

                var src_vec = rebind[self.element_type](
                    SIMD[size=src_element_size].load[
                        alignment = alignof[self.element_type]()
                    ](other.ptr, src_idx)
                )

                @parameter
                for e_i in range(src_element_size):
                    alias dst_idx = make_layout(
                        self.element_layout, self.layout
                    )(i * dst_element_size + e_i)
                    Scalar.store(self.ptr, dst_idx, src_vec[e_i])

        # Vector write scalar reads.
        elif (
            other_element_layout.rank() == 1
            and other_element_layout.stride[0] != 1
            and self.element_layout.rank() == 1
            and self.element_layout.stride[0] == 1
        ):

            @parameter
            for i in range(dst_size):
                alias dst_idx = make_layout(self.element_layout, self.layout)(
                    i * dst_element_size
                )

                var src_vec = self.element_type()

                @parameter
                for e_i in range(src_element_size):
                    alias src_idx = make_layout(
                        other_element_layout, other.layout
                    )(i * src_element_size + e_i)
                    src_vec[e_i] = Scalar.load(other.ptr, src_idx)

                SIMD[size = self.element_size].store(self.ptr, dst_idx, src_vec)

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
            for i in range(dst_size):
                # Offset to the current element.
                alias src_offset = other_layout(i)
                alias dst_offset = self.layout(i)
                alias num_copies = self.element_layout.shape[0].value()
                alias vec_width = self.element_layout.shape[1].value()

                @parameter
                for j in range(num_copies):
                    alias src_idx = src_offset + other_element_layout(j)
                    alias dst_idx = dst_offset + self.element_layout(j)

                    var src_vec = SIMD[size=vec_width].load[
                        alignment = alignof[SIMD[dtype, vec_width]]()
                    ](other.ptr, src_idx).cast[dtype]()

                    SIMD[size=vec_width].store[
                        alignment = alignof[SIMD[dtype, vec_width]](),
                    ](self.ptr, dst_idx, src_vec)
        else:

            @parameter
            for i in range(dst_size * dst_element_size):
                alias src_idx = make_layout(other.element_layout, other_layout)(
                    i
                )
                alias dst_idx = make_layout(self.element_layout, self.layout)(i)

                @parameter
                if other.masked:
                    alias idx = idx2crd(
                        dst_idx, self.layout.shape, self.layout.stride
                    )
                    var can_access_src = other._is_not_masked_elemenet[idx]()
                    if can_access_src:
                        self.ptr[dst_idx] = other.ptr[src_idx]
                elif Self.masked:
                    alias idx = idx2crd(
                        src_idx, self.layout.shape, self.layout.stride
                    )
                    var can_access_dst = self._is_not_masked_elemenet[idx]()
                    if can_access_dst:
                        self.ptr[dst_idx] = other.ptr[src_idx]
                else:
                    self.ptr[dst_idx] = other.ptr[src_idx]

    @always_inline
    fn copy_from_async[
        src_layout: Layout,
        src_addr_space: AddressSpace,
        src_element_layout: Layout,
        src_mask: Bool,
    ](
        self,
        src: LayoutTensor[
            dtype,
            src_layout,
            address_space=src_addr_space,
            element_layout=src_element_layout,
            masked=src_mask,
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
            for i in range(dst_size):
                alias src_idx = make_layout(src.element_layout, src_layout)(
                    i * src_element_size
                )
                alias dst_idx = make_layout(self.element_layout, self.layout)(
                    i * dst_element_size
                )

                async_copy[element_size_bytes](
                    src_ptr + src_idx, dst_ptr + dst_idx
                )

        else:

            @parameter
            for i in range(dst_size * dst_element_size):
                alias src_idx = make_layout(src.element_layout, src_layout)(i)
                alias dst_idx = make_layout(self.element_layout, self.layout)(i)

                async_copy[4](src_ptr + src_idx, dst_ptr + dst_idx)

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

        @parameter
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
                for idx in range(Self.element_size):
                    alias element_offset = self.element_layout(idx)
                    vec[idx] = Scalar.load(
                        self.ptr, vec_offset + element_offset
                    )

                print(vec)


fn stack_allocation_like[
    layout: Layout,
    dtype: DType,
    *,
    address_space: AddressSpace,
    masked: Bool,
    target_address_space: AddressSpace = AddressSpace.GENERIC,
](
    in_tensor: LayoutTensor[
        dtype, layout, address_space=address_space, masked=masked
    ]
) -> LayoutTensor[
    dtype, layout, address_space=target_address_space, masked=masked
]:
    return LayoutTensor[
        dtype, layout, address_space=target_address_space, masked=masked
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
    res_masked: Bool,
    lhs_masked: Bool,
    rhs_masked: Bool,
](
    res: LayoutTensor[
        dtype, res_layout, address_space=res_address_space, masked=res_masked
    ],
    lhs: LayoutTensor[
        _, lhs_layout, address_space=lhs_address_space, masked=lhs_masked
    ],
    rhs: LayoutTensor[
        _, rhs_layout, address_space=rhs_address_space, masked=rhs_masked
    ],
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
    src_mask: Bool,
    dst_mask: Bool,
    swizzle: OptionalReg[_swizzle_signature] = None,
](
    dst: LayoutTensor[
        dtype,
        dst_layout,
        address_space = _GPUAddressSpace.SHARED,
        element_layout=dst_element_layout,
        masked=dst_mask,
    ],
    src: LayoutTensor[
        dtype,
        src_layout,
        address_space = _GPUAddressSpace.GENERIC,
        element_layout=src_element_layout,
        masked=src_mask,
    ],
):
    var src_framgents = src.distribute[src_thread_layout](ThreadIdx.x())
    var dst_framgents = dst.distribute[dst_thread_layout, swizzle=swizzle](
        ThreadIdx.x()
    )
    dst_framgents.copy_from(src_framgents)


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
    src_mask: Bool,
    dst_mask: Bool,
    swizzle: OptionalReg[_swizzle_signature] = None,
](
    dst: LayoutTensor[
        dtype,
        dst_layout,
        address_space = _GPUAddressSpace.SHARED,
        element_layout=dst_element_layout,
        masked=dst_mask,
    ],
    src: LayoutTensor[
        dtype,
        src_layout,
        address_space = _GPUAddressSpace.GENERIC,
        element_layout=src_element_layout,
        masked=src_mask,
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
        src_mask,
        dst_mask,
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
    src_mask: Bool,
    dst_mask: Bool,
    swizzle: OptionalReg[_swizzle_signature] = None,
](
    dst: LayoutTensor[
        dtype,
        dst_layout,
        address_space = _GPUAddressSpace.SHARED,
        element_layout=dst_element_layout,
        masked=dst_mask,
    ],
    src: LayoutTensor[
        dtype,
        src_layout,
        address_space = _GPUAddressSpace.GENERIC,
        element_layout=src_element_layout,
        masked=src_mask,
    ],
):
    var src_framgents = src.distribute[src_thread_layout](ThreadIdx.x())
    var dst_framgents = dst.distribute[dst_thread_layout, swizzle=swizzle](
        ThreadIdx.x()
    )
    dst_framgents.copy_from_async(src_framgents)


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
    src_mask: Bool,
    dst_mask: Bool,
    swizzle: OptionalReg[_swizzle_signature] = None,
](
    dst: LayoutTensor[
        dtype,
        dst_layout,
        address_space = _GPUAddressSpace.SHARED,
        element_layout=dst_element_layout,
        masked=dst_mask,
    ],
    src: LayoutTensor[
        dtype,
        src_layout,
        address_space = _GPUAddressSpace.GENERIC,
        element_layout=src_element_layout,
        masked=src_mask,
    ],
):
    copy_dram_to_sram_async[
        src_layout,
        dst_layout,
        dtype,
        thread_layout,
        thread_layout,
        src_element_layout,
        dst_element_layout,
        src_mask,
        dst_mask,
        swizzle,
    ](dst, src)


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
    src_mask: Bool,
    dst_mask: Bool,
    axis: Optional[Int] = None,
](
    dst: LayoutTensor[
        dtype,
        dst_layout,
        address_space = _GPUAddressSpace.GENERIC,
        element_layout=dst_element_layout,
        masked=dst_mask,
    ],
    src: LayoutTensor[
        dtype,
        src_layout,
        address_space = _GPUAddressSpace.SHARED,
        element_layout=src_element_layout,
        masked=src_mask,
    ],
):
    @parameter
    if axis:
        var src_fragments = src.distribute[
            src_warp_layout, axis = axis._value_copy()
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
    src_mask: Bool,
    dst_mask: Bool,
](
    dst: LayoutTensor[
        dtype,
        dst_layout,
        address_space = _GPUAddressSpace.GENERIC,
        element_layout=dst_element_layout,
        masked=dst_mask,
    ],
    src: LayoutTensor[
        dtype,
        src_layout,
        address_space = _GPUAddressSpace.GENERIC,
        element_layout=src_element_layout,
        masked=src_mask,
    ],
):
    var dst_framgents = dst.distribute[dst_thread_layout](ThreadIdx.x())
    dst_framgents.copy_from(src)


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

    var ptr: DTypePointer[type, address_space]
    var offset: Int
    var stride: Int
    var bound: Int

    @always_inline
    fn __init__(
        inout self,
        ptr: DTypePointer[type, address_space],
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
    fn next(self) -> Self:
        return self.next(1)
