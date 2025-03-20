# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import OptionalReg
from sys import alignof, sizeof

from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from gpu.id import thread_idx
from gpu.memory import CacheEviction, Fill, async_copy
from layout import Layout, LayoutTensor
from layout.int_tuple import depth
from layout.layout import make_layout
from memory.pointer import AddressSpace, _GPUAddressSpace

from utils import IndexList, StaticTuple

alias _swizzle_signature = fn[type: DType] (Scalar[type]) -> Scalar[type]


# TileMask holds information collected by composed tile operations to
# determine per dim mask.
# Note: The reason we want per-dim mask is because vectorized `non-scalar`
# elements are n-d, and it can be OOB only with respect to a specific axis.
#
@value
struct TileMask[
    rank: Int,
    element_size: IndexList[rank] = IndexList[rank](1),
    element_stride: IndexList[rank] = IndexList[rank](1),
]:
    var max_dim: IndexList[rank]
    var offset: IndexList[rank]

    fn __init__(
        mut self,
        max_dim: IndexList[rank],
        offset: IndexList[rank] = IndexList[rank](0),
    ):
        self.max_dim = max_dim
        self.offset = offset

    # Returns a Tuple[rank] where particular axis is true if the tile can be
    # accessed at the given `point` at this axis.
    #
    @always_inline
    fn access_mask(self, point: IndexList[rank]) -> StaticTuple[Bool, rank]:
        var mask = StaticTuple[Bool, rank]()

        @parameter
        for axis in range(rank):

            @parameter
            if element_size[axis] == 1:
                mask[axis] = (
                    self.offset[axis] + point[axis] * element_stride[axis]
                ) < self.max_dim[axis]
            else:
                mask[axis] = (
                    self.offset[axis]
                    + point[axis] * element_size[axis] * element_stride[axis]
                    + element_size[axis]
                ) < self.max_dim[axis]

        return mask

    # Returns the element size can be accessed.
    #
    @always_inline
    fn access_size(
        self, point: IndexList[rank], dim_mask: StaticTuple[Bool, rank]
    ) -> IndexList[rank]:
        var size = IndexList[rank]()

        @parameter
        for i in range(rank):
            if dim_mask[i]:
                size[i] = element_size[i]
            else:
                var start_index = self.offset[i] + point[i] * element_size[
                    i
                ] * element_stride[i]
                size[i] = max(0, self.max_dim[i] - start_index)

        return size


# Computes the mask resulting tiling buffer with the `tile sizes`.
#
@always_inline("nodebug")
fn _tile_mask[
    *tile_sizes: Dim,
    rank: Int,
    __sizes: IndexList[rank] = IndexList[rank](1),
    __element_stride: IndexList[rank] = IndexList[rank](1),
](
    shape: IndexList[rank],
    tile_coords: IndexList[rank],
    out result: TileMask[rank, __sizes, __element_stride],
):
    var tile_offset = IndexList[rank]()

    @parameter
    for i in range(rank):
        tile_offset[i] = tile_sizes[i].get() * tile_coords[i]

    return __type_of(result)(shape, tile_offset)


@always_inline("nodebug")
fn _to_static_tuple[rank: Int](sizes: VariadicList[Int]) -> IndexList[rank]:
    var res = IndexList[rank]()

    @parameter
    for i in range(rank):
        res[i] = sizes[i]

    return res


# Computes the mask resulting vectorizing buffer with the `sizes`.
#
@always_inline("nodebug")
fn _vectorize_mask[
    rank: Int,
    sizes: IndexList[rank],
    element_stride: IndexList[rank],
    mask_sizes: IndexList[rank],
](mask: TileMask[rank, mask_sizes, element_stride]) -> TileMask[
    rank, sizes, element_stride
]:
    var res = TileMask[rank, sizes, element_stride](mask.max_dim, mask.offset)
    return res


# Returns the shaep of the `thread_layout` as tuple.
#
@always_inline("nodebug")
fn _get_shape_as_tuple[
    rank: Int,
](thread_layout: Layout) -> IndexList[rank]:
    var res = IndexList[rank]()

    @parameter
    for i in range(rank):
        res[i] = Int(thread_layout.shape[i])

    return res


# Computes the mask resulting distributing to `thread_layout`.
#
@always_inline("nodebug")
fn _distribute_mask[
    thread_layout: Layout,
    rank: Int,
    element_size: IndexList[rank],
    element_stride: IndexList[rank],
    __element_stride: IndexList[rank] = _get_shape_as_tuple[rank](
        thread_layout
    ),
](
    mask: TileMask[rank, element_size, element_stride],
    thread_id: Int,
) -> TileMask[rank, element_size, __element_stride]:
    var res = TileMask[rank, element_size, __element_stride](
        mask.max_dim, mask.offset
    )

    @parameter
    for i in range(rank):
        alias shape_i = Int(thread_layout.shape[i])
        alias stride_i = Int(thread_layout.stride[i])
        var thread_coord_i = (thread_id // stride_i) % shape_i
        res.offset[i] += thread_coord_i

    return res


# Returns the shape of distribute `thread_layout` into `shape`.
#
@always_inline("nodebug")
fn _distribute_shape[thread_layout: Layout](shape: DimList) -> DimList:
    constrained[
        thread_layout.rank() <= 3,
        "_distribute_shape requires thread_layout <= 3",
    ]()

    var res = StaticTuple[Dim][thread_layout.rank()]()

    @parameter
    for i in range(thread_layout.rank()):
        if shape.at[i]().is_dynamic():
            res[i] = Dim()
        else:
            res[i] = shape.at[i]() // Int(thread_layout.shape[i])

    if thread_layout.rank() == 1:
        return DimList(res[0])
    elif thread_layout.rank() == 2:
        return DimList(res[0], res[1])
    elif thread_layout.rank() == 3:
        return DimList(res[0], res[1], res[2])
    return DimList()


# Distribute thread_layout and returns the fragments of `thread_id`.
#
@always_inline("nodebug")
fn distribute[
    dtype: DType,
    rank: Int,
    shape: DimList,
    thread_layout: Layout,
    _result_shape: DimList = _distribute_shape[thread_layout](shape),
    swizzle: OptionalReg[_swizzle_signature] = None,
    element_size: Int = 1,
](buff: NDBuffer[dtype, rank, shape], thread_id: Int) -> NDBuffer[
    dtype, rank, _result_shape
]:
    constrained[
        depth(thread_layout.shape) == 1,
        "distribute threads to NDBuffer only supports depth-1 thread layouts",
    ]()

    var res_strides = IndexList[rank]()
    var res_shape = IndexList[rank]()

    @parameter
    for i in range(rank):
        alias thread_shape_i = Int(thread_layout.shape[i])
        res_shape[i] = buff.dim[i]() // thread_shape_i
        res_strides[i] = buff.stride[i]() * thread_shape_i

    var thread_offset: Int32 = 0

    @parameter
    for i in range(rank):
        alias shape_i = Int(thread_layout.shape[i])
        alias stride_i = Int(thread_layout.stride[i])
        var thread_coords_i = (thread_id // stride_i) % shape_i
        thread_offset += thread_coords_i * buff.stride[i]()

    @parameter
    if swizzle:
        alias swizzle_fn = swizzle.value()
        thread_offset = (
            swizzle_fn[DType.int32](thread_offset // element_size)
            * element_size
        )

    var res = NDBuffer[dtype, rank, _result_shape](
        buff.data.offset(Int(thread_offset)),
        dynamic_shape=res_shape,
        dynamic_stride=res_strides,
    )
    return res


# FIXME: Move to a shared utility.
# Returns the size of variadic integer parameters.
#
@always_inline("nodebug")
fn _get_len[*var_int: Int]() -> Int:
    return __mlir_op.`pop.variadic.size`(var_int)


@always_inline("nodebug")
fn _vectorize_shape[*sizes: Int](shape: DimList) -> DimList:
    alias rank = _get_len[*sizes]()

    constrained[
        rank <= 3,
        "_vectorize_shape vector sizes <= 3",
    ]()

    var res = StaticTuple[Dim, rank]()

    @parameter
    for i in range(rank):
        alias size_i = sizes[i]

        if shape.at[i]().is_dynamic():
            res[i] = Dim()
        else:
            res[i] = shape.at[i]() // size_i

    @parameter
    if rank == 1:
        return DimList(res[0])
    elif rank == 2:
        return DimList(res[0], res[1])
    elif rank == 3:
        return DimList(res[0], res[1], res[2])
    return DimList()


@always_inline("nodebug")
fn _to_static_tuple[*sizes: Int, rank: Int]() -> IndexList[rank]:
    var vals = IndexList[rank]()

    @parameter
    for i in range(rank):
        vals[i] = sizes[i]

    return vals


# Stores the layout of the vectorized buffer element.
#
@value
struct ElementLayout[rank: Int, shape: IndexList[rank]](
    CollectionElement,
    Stringable,
    Writable,
):
    var stride: IndexList[rank]

    fn __init__(out self):
        self.stride = IndexList[rank]()

    fn __init__(out self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        writer.write(shape, ":", self.stride)


# Returns the linear index of an element, this is equivalent to concat
# the element layout and the buffer layout
@always_inline("nodebug")
fn _get_element_idx[
    rank: Int,
    dtype: DType,
    shape: DimList,
    element_shape: IndexList[rank],
](
    linear_coord: Int,
    buff: NDBuffer[dtype, rank, shape],
    element_layout: ElementLayout[rank, element_shape],
) -> Int:
    var result = 0
    var curr_linear_crd = linear_coord

    # evaluate according to
    # iterate over outer most
    @parameter
    for i in range(rank):
        result += (
            curr_linear_crd % element_layout.shape[i]
        ) * element_layout.stride[i]
        curr_linear_crd = curr_linear_crd // element_layout.shape[i]

    @parameter
    for i in range(rank):
        result += (curr_linear_crd % buff.dim[i]()) * buff.stride[i]()
        curr_linear_crd = curr_linear_crd // buff.dim[i]()

    return result


@always_inline("nodebug")
fn _get_element_idx[
    rank: Int,
    dtype: DType,
    shape: DimList,
](linear_coord: Int, buff: NDBuffer[dtype, rank, shape]) -> Int:
    var result = 0
    var curr_linear_crd = linear_coord

    @parameter
    for i in range(rank):
        result += (curr_linear_crd % buff.dim[i]()) * buff.stride[i]()
        curr_linear_crd = curr_linear_crd // buff.dim[i]()
    return result


@always_inline("nodebug")
fn _get_element_idx[
    rank: Int,
    element_shape: IndexList[rank],
](
    linear_coord: Int,
    element_layout: ElementLayout[rank, element_shape],
) -> Int:
    var result = 0
    var curr_linear_crd = linear_coord

    # evaluate according to
    # iterate over outer most
    @parameter
    for i in range(rank):
        result += (
            curr_linear_crd % element_layout.shape[i]
        ) * element_layout.stride[i]
        curr_linear_crd = curr_linear_crd // element_layout.shape[i]

    return result


# Vectorizes buffer and returns the vecrtorized buffer and its dynamic layout.
#
@always_inline("nodebug")
fn vectorize[
    *sizes: Int,
    dtype: DType,
    rank: Int,
    shape: DimList,
    _res_shape: DimList = _vectorize_shape[*sizes](shape),
](buff: NDBuffer[dtype, rank, shape, *_]) -> Tuple[
    NDBuffer[
        dtype,
        rank,
        shape=_res_shape,
        strides = DimList.create_unknown[rank](),
        address_space = buff.address_space,
    ],
    ElementLayout[rank, _to_static_tuple[*sizes, rank=rank]()],
]:
    var buff_shape = IndexList[rank]()
    var buff_stride = IndexList[rank]()

    var element_layout = ElementLayout[
        rank, _to_static_tuple[*sizes, rank=rank]()
    ]()

    @parameter
    for i in range(rank):
        element_layout.stride[i] = buff.stride[i]()
        buff_shape[i] = buff.dim[i]() // sizes[i]
        buff_stride[i] = buff.stride[i]() * sizes[i]

    return Tuple(
        NDBuffer[
            dtype,
            rank,
            shape=_res_shape,
            strides = DimList.create_unknown[rank](),
            address_space = buff.address_space,
        ](buff.data, dynamic_shape=buff_shape, dynamic_stride=buff_stride),
        element_layout,
    )


@always_inline("nodebug")
fn _copy_nd_buffer_to_layout_tensor[
    src_rank: Int,
    dtype: DType,
    layout: Layout,
    shape: DimList,
    buff_element_layout_shape: IndexList[src_rank],
    *,
    is_async: Bool = False,
    fill: OptionalReg[Scalar[dtype]] = None,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
](
    dst: LayoutTensor[
        mut=True,
        dtype,
        layout,
        *_, **_,
    ],
    src: NDBuffer[dtype, src_rank, shape],
    buff_element_layout: ElementLayout[src_rank, buff_element_layout_shape],
):
    alias num_elements = dst.layout.size()
    alias dst_rank = layout.rank()
    alias tensor_element_layout = dst.element_layout
    constrained[src_rank == dst_rank, "src and dst should have same rank"]()

    # 1d-vector load/store
    @parameter
    if (
        tensor_element_layout.rank() == 1
        and tensor_element_layout.stride[0] == 1
        and dst.element_size != 1
    ):
        constrained[
            tensor_element_layout.shape[0] == buff_element_layout_shape[1],
            "LayoutTensor element shape != buffer element shape",
        ]()
        constrained[buff_element_layout_shape[0] == 1, "Expecting row vector"]()

        alias vec_size = Int(tensor_element_layout.shape[0])
        alias alignment = alignof[dst.element_type]()

        @parameter
        for i in range(num_elements):
            alias dst_idx = make_layout(tensor_element_layout, dst.layout)(
                i * vec_size
            )
            var src_idx = _get_element_idx(
                i * vec_size, src, buff_element_layout
            )

            @parameter
            if is_async:
                alias element_size_bytes = vec_size * sizeof[dtype]()
                var src_ptr = src.data.address_space_cast[
                    _GPUAddressSpace.GLOBAL
                ]()
                var dst_ptr = dst.ptr.address_space_cast[
                    _GPUAddressSpace.SHARED
                ]()
                async_copy[
                    element_size_bytes,
                    fill=fill,
                    eviction_policy=eviction_policy,
                ](src_ptr + src_idx, dst_ptr + dst_idx)
            else:
                var src_element = src.data.offset(src_idx).load[
                    width=vec_size, alignment=alignment
                ]()
                dst.ptr.store[alignment=alignment](dst_idx, src_element)

    # 2d-vector load/store
    elif (
        tensor_element_layout.rank() == 2
        and tensor_element_layout.stride[1] == 1
    ):
        alias num_copies = tensor_element_layout.shape[0].value()
        alias vec_width = tensor_element_layout.shape[1].value()

        @parameter
        for i in range(num_elements):
            alias dst_offset = layout(i)
            var src_offset = _get_element_idx(i, src)

            @parameter
            for j in range(num_copies):
                alias dst_idx = dst_offset + tensor_element_layout(j)
                var src_idx = src_offset + _get_element_idx(
                    j, buff_element_layout
                )

                @parameter
                if is_async:
                    alias element_size_bytes = vec_width * sizeof[dtype]()
                    var src_ptr = src.data.address_space_cast[
                        _GPUAddressSpace.GLOBAL
                    ]()
                    var dst_ptr = dst.ptr.address_space_cast[
                        _GPUAddressSpace.SHARED
                    ]()
                    async_copy[
                        element_size_bytes,
                        fill=fill,
                        eviction_policy=eviction_policy,
                    ](src_ptr + src_idx, dst_ptr + dst_idx)
                else:
                    var src_vec = src.data.load[
                        width=vec_width,
                        alignment = alignof[SIMD[dtype, vec_width]](),
                    ](src_idx).cast[dtype]()

                    dst.ptr.store[
                        alignment = alignof[SIMD[dtype, vec_width]](),
                    ](dst_idx, src_vec)

    # Scalar case.
    else:

        @parameter
        for i in range(num_elements * dst.element_size):
            alias dst_idx = make_layout(tensor_element_layout, dst.layout)(i)
            var src_idx = _get_element_idx(i, src, buff_element_layout)

            @parameter
            if is_async:
                var src_ptr = src.data.address_space_cast[
                    _GPUAddressSpace.GLOBAL
                ]()
                var dst_ptr = dst.ptr.address_space_cast[
                    _GPUAddressSpace.SHARED
                ]()
                async_copy[4, fill=fill, eviction_policy=eviction_policy](
                    src_ptr + src_idx, dst_ptr + dst_idx
                )
            else:
                dst.ptr[dst_idx] = src.data[src_idx]


@always_inline("nodebug")
fn _copy_nd_buffer_to_layout_tensor_masked[
    src_rank: Int,
    dtype: DType,
    layout: Layout,
    shape: DimList,
    buff_element_layout_shape: IndexList[src_rank],
    mask_rank: Int,
    mask_element_size: IndexList[mask_rank],
    mask_element_stride: IndexList[mask_rank],
    *,
    is_async: Bool = False,
    fill: OptionalReg[Scalar[dtype]] = None,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
](
    dst: LayoutTensor[
        mut=True,
        dtype,
        layout,
        *_, **_,
    ],
    src: NDBuffer[dtype, src_rank, shape],
    buff_element_layout: ElementLayout[src_rank, buff_element_layout_shape],
    tile_mask: TileMask[mask_rank, mask_element_size, mask_element_stride],
):
    alias num_elements = dst.layout.size()
    alias dst_rank = layout.rank()
    alias tensor_element_layout = dst.element_layout
    constrained[src_rank == dst_rank, "src and dst should have same rank"]()
    constrained[
        mask_rank == dst_rank, "mask_rank and dst should have same rank"
    ]()

    constrained[mask_rank == 2, "Masking is only supported for rank-2 inputs"]()

    constrained[
        mask_element_size[0] * mask_element_size[1] == 1,
        "Only scalar element masksing is supported",
    ]()

    # 1d-vector load/store
    @parameter
    if (
        tensor_element_layout.rank() == 1
        and tensor_element_layout.stride[0] == 1
        and dst.element_size != 1
    ):
        constrained[
            tensor_element_layout.shape[0] == buff_element_layout_shape[1],
            "LayoutTensor element shape != buffer element shape",
        ]()
        constrained[buff_element_layout_shape[0] == 1, "Expecting row vector"]()

        alias vec_size = Int(tensor_element_layout.shape[0])
        alias alignment = alignof[dst.element_type]()

        @parameter
        for i in range(num_elements):
            alias dst_idx = make_layout(tensor_element_layout, dst.layout)(
                i * vec_size
            )
            var src_idx = _get_element_idx(
                i * vec_size, src, buff_element_layout
            )

            @parameter
            if is_async:
                alias element_size_bytes = vec_size * sizeof[dtype]()
                var src_ptr = src.data.address_space_cast[
                    _GPUAddressSpace.GLOBAL
                ]()
                var dst_ptr = dst.ptr.address_space_cast[
                    _GPUAddressSpace.SHARED
                ]()
                async_copy[
                    element_size_bytes,
                    fill=fill,
                    eviction_policy=eviction_policy,
                ](src_ptr + src_idx, dst_ptr + dst_idx)
            else:
                var src_element = src.data.offset(src_idx).load[
                    width=vec_size, alignment=alignment
                ]()
                dst.ptr.store[alignment=alignment](dst_idx, src_element)

    # 2d-vector load/store
    elif (
        tensor_element_layout.rank() == 2
        and tensor_element_layout.stride[1] == 1
    ):
        alias num_copies = tensor_element_layout.shape[0].value()
        alias vec_width = tensor_element_layout.shape[1].value()

        @parameter
        for i in range(num_elements):
            alias dst_offset = layout(i)
            var src_offset = _get_element_idx(i, src)

            @parameter
            for j in range(num_copies):
                alias dst_idx = dst_offset + tensor_element_layout(j)
                var src_idx = src_offset + _get_element_idx(
                    j, buff_element_layout
                )

                @parameter
                if is_async:
                    alias element_size_bytes = vec_width * sizeof[dtype]()
                    var src_ptr = src.data.address_space_cast[
                        _GPUAddressSpace.GLOBAL
                    ]()
                    var dst_ptr = dst.ptr.address_space_cast[
                        _GPUAddressSpace.SHARED
                    ]()
                    async_copy[
                        element_size_bytes,
                        fill=fill,
                        eviction_policy=eviction_policy,
                    ](src_ptr + src_idx, dst_ptr + dst_idx)
                else:
                    var src_vec = src.data.load[
                        width=vec_width,
                        alignment = alignof[SIMD[dtype, vec_width]](),
                    ](src_idx).cast[dtype]()

                    dst.ptr.store[
                        alignment = alignof[SIMD[dtype, vec_width]](),
                    ](dst_idx, src_vec)

    # Scalar case.
    else:

        @parameter
        for i in range(num_elements * dst.element_size):
            alias dst_idx = make_layout(tensor_element_layout, dst.layout)(i)
            var src_idx = _get_element_idx(i, src, buff_element_layout)

            # Evaluate the mask, skip OOB element copies
            alias dim_0_shape = Int(dst.layout.shape[0])
            var dim_0 = i % dim_0_shape
            var dim_1 = i // dim_0_shape
            var mask_val = tile_mask.access_mask((dim_0, dim_1))
            var can_access = mask_val[0] and mask_val[1]
            if not can_access:
                continue

            @parameter
            if is_async:
                var src_ptr = src.data.address_space_cast[
                    _GPUAddressSpace.GLOBAL
                ]()
                var dst_ptr = dst.ptr.address_space_cast[
                    _GPUAddressSpace.SHARED
                ]()
                async_copy[4, fill=fill, eviction_policy=eviction_policy](
                    src_ptr + src_idx, dst_ptr + dst_idx
                )
            else:
                dst.ptr[dst_idx] = src.data[src_idx]


@always_inline("nodebug")
fn _copy_layout_tensor_to_nd_buffer[
    dst_rank: Int,
    dtype: DType,
    layout: Layout,
    shape: DimList,
    buff_element_layout_shape: IndexList[dst_rank],
](
    dst: NDBuffer[dtype, dst_rank, shape],
    buff_element_layout: ElementLayout[dst_rank, buff_element_layout_shape],
    src: LayoutTensor[
        dtype,
        layout,
        *_, **_,
    ],
):
    alias src_rank = layout.rank()
    alias tensor_element_layout = src.element_layout
    alias num_elements = src.layout.size()
    constrained[src_rank == dst_rank, "src and dst should have same rank"]()

    # 1d-vector load/store
    @parameter
    if (
        tensor_element_layout.rank() == 1
        and tensor_element_layout.stride[0] == 1
        and src.element_size != 1
    ):
        constrained[
            tensor_element_layout.shape[0] == buff_element_layout_shape[1],
            "LayoutTensor element shape != buffer element shape",
        ]()
        constrained[buff_element_layout_shape[0] == 1, "Expecting row vector"]()

        alias vec_size = Int(tensor_element_layout.shape[0])
        alias alignment = alignof[src.element_type]()

        @parameter
        for i in range(num_elements):
            var dst_idx = _get_element_idx(
                i * vec_size, dst, buff_element_layout
            )
            alias src_idx = make_layout(tensor_element_layout, src.layout)(
                i * vec_size
            )

            var src_element = src.ptr.offset(src_idx).load[
                width=vec_size, alignment=alignment
            ]()
            dst.data.store[alignment=alignment](dst_idx, src_element)

    # 2d-vector load/store
    elif (
        tensor_element_layout.rank() == 2
        and tensor_element_layout.stride[1] == 1
    ):
        alias num_copies = tensor_element_layout.shape[0].value()
        alias vec_width = tensor_element_layout.shape[1].value()

        @parameter
        for i in range(num_elements):
            # Offset to the current element.
            var dst_offset = _get_element_idx(i, dst)

            alias src_offset = layout(i)

            @parameter
            for j in range(num_copies):
                var dst_idx = dst_offset + _get_element_idx(
                    j, buff_element_layout
                )
                alias src_idx = src_offset + tensor_element_layout(j)

                var src_vec = src.ptr.load[
                    width=vec_width,
                    alignment = alignof[SIMD[dtype, vec_width]](),
                ](src_idx).cast[dtype]()

                dst.data.store[alignment = alignof[SIMD[dtype, vec_width]]()](
                    dst_idx, src_vec
                )

    # Scalar case.
    else:

        @parameter
        for i in range(num_elements * src.element_size):
            alias src_idx = make_layout(tensor_element_layout, src.layout)(i)
            var dst_idx = _get_element_idx(i, dst, buff_element_layout)
            dst.data[dst_idx] = src.ptr[src_idx]


@always_inline
fn _copy_layout_tensor_to_nd_buffer_masked[
    dst_rank: Int,
    mask_rank: Int,
    dtype: DType,
    layout: Layout,
    shape: DimList,
    buff_element_layout_shape: IndexList[dst_rank],
    mask_element_size: IndexList[mask_rank],
    mask_element_stride: IndexList[mask_rank],
](
    dst: NDBuffer[dtype, dst_rank, shape],
    buff_element_layout: ElementLayout[dst_rank, buff_element_layout_shape],
    src: LayoutTensor[
        dtype,
        layout,
        *_, **_,
    ],
    tile_mask: TileMask[mask_rank, mask_element_size, mask_element_stride],
):
    alias num_elements = src.layout.size()
    alias src_rank = layout.rank()
    alias tensor_element_layout = src.element_layout
    constrained[src_rank == dst_rank, "src and dst should have same rank"]()

    constrained[
        mask_rank == dst_rank, "mask_rank and dst should have same rank"
    ]()

    constrained[mask_rank == 2, "Masking is only supported for rank-2 inputs"]()

    constrained[
        mask_element_size[0] * mask_element_size[1] == 1,
        "Only scalar element masksing is supported",
    ]()

    # 1d-vector load/store
    @parameter
    if (
        tensor_element_layout.rank() == 1
        and tensor_element_layout.stride[0] == 1
        and src.element_size != 1
    ):
        constrained[
            tensor_element_layout.shape[0] == buff_element_layout_shape[1],
            "LayoutTensor element shape != buffer element shape",
        ]()
        constrained[buff_element_layout_shape[0] == 1, "Expecting row vector"]()

        alias vec_size = Int(tensor_element_layout.shape[0])
        alias alignment = alignof[src.element_type]()

        @parameter
        for i in range(num_elements):
            var dst_idx = _get_element_idx(
                i * vec_size, dst, buff_element_layout
            )
            alias src_idx = make_layout(tensor_element_layout, src.layout)(
                i * vec_size
            )

            var src_element = src.ptr.offset(src_idx).load[
                width=vec_size, alignment=alignment
            ]()
            dst.data.offset(dst_idx).store[alignment=alignment](src_element)

    # 2d-vector load/store
    elif (
        tensor_element_layout.rank() == 2
        and tensor_element_layout.stride[1] == 1
    ):
        alias num_copies = tensor_element_layout.shape[0].value()
        alias vec_width = tensor_element_layout.shape[1].value()

        @parameter
        for i in range(num_elements):
            # Offset to the current element.
            var dst_offset = _get_element_idx(i, dst)

            alias src_offset = layout(i)

            @parameter
            for j in range(num_copies):
                var dst_idx = dst_offset + _get_element_idx(
                    j, buff_element_layout
                )
                alias src_idx = src_offset + tensor_element_layout(j)

                var src_vec = src.ptr.load[
                    width=vec_width,
                    alignment = alignof[SIMD[dtype, vec_width]](),
                ](src_idx).cast[dtype]()

                dst.data.store[alignment = alignof[SIMD[dtype, vec_width]]()](
                    dst_idx, src_vec
                )

    # Scalar case.
    else:

        @parameter
        for i in range(num_elements * src.element_size):
            # Evaluate the mask, skip OOB element copies
            alias dim_0_shape = Int(src.layout.shape[0])
            var dim_0 = i % dim_0_shape
            var dim_1 = i // dim_0_shape
            var mask_val = tile_mask.access_mask((dim_0, dim_1))
            var can_access = mask_val[0] and mask_val[1]
            if not can_access:
                continue

            alias src_idx = make_layout(tensor_element_layout, src.layout)(i)
            var dst_idx = _get_element_idx(i, dst, buff_element_layout)
            dst.data[dst_idx] = src.ptr[src_idx]


# Copies an nd-buffer fragment to `thread_id` thread local LayoutTensor element
# where each element of the fragment is originally distributed by `thread_layout`.
#
@always_inline("nodebug")
fn copy_from_nd_buffer[
    dtype: DType,
    dst_data_layout: Layout, //,
    thread_layout: Layout,
    is_async: Bool = False,
    swizzle: OptionalReg[_swizzle_signature] = None,
](
    dst_thread_local: LayoutTensor[
        mut=True,
        dtype,
        dst_data_layout,
        *_, **_,
    ],
    src: NDBuffer[dtype, *_],
    thread_id: Int,
):
    alias dst_rank = dst_data_layout.rank()
    alias dst_element_layout = dst_thread_local.element_layout
    # FIXME: Relax this to support any ranked data and thread layouts.
    constrained[src.rank == dst_rank, "src and dst should have same rank"]()
    constrained[dst_rank == 2, "Only rank-2 layouts is supported for now."]()

    constrained[
        dst_thread_local.element_layout.rank() == 1
        or dst_thread_local.element_layout.rank() == 2,
        "Only rank-1, rank-2 vectoriztion is supported",
    ]()

    alias threads_layout_rank = thread_layout.rank()
    constrained[
        threads_layout_rank == dst_rank,
        "thread and data layout should have the same rank",
    ]()

    @parameter
    if dst_element_layout.rank() == 1:
        var src_vectorized = vectorize[1, Int(dst_element_layout.shape[0])](src)
        var src_vectorized_buffer = src_vectorized[0]
        var src_element_layout = src_vectorized[1]
        alias element_size = Int(dst_element_layout.shape[0])
        var src_thread_local = distribute[
            thread_layout=thread_layout,
            swizzle=swizzle,
            element_size=element_size,
        ](src_vectorized_buffer, thread_id)
        _copy_nd_buffer_to_layout_tensor[is_async=is_async](
            dst_thread_local, src_thread_local, src_element_layout
        )
    elif dst_element_layout.rank() == 2:
        var src_vectorized = vectorize[
            Int(dst_element_layout.shape[0]),
            Int(dst_element_layout.shape[1]),
        ](src)
        var src_vectorized_buffer = src_vectorized[0]
        var src_element_layout = src_vectorized[1]
        alias element_size = Int(dst_element_layout.shape[1])
        var src_thread_local = distribute[
            thread_layout=thread_layout,
            swizzle=swizzle,
            element_size=element_size,
        ](src_vectorized_buffer, thread_id)
        _copy_nd_buffer_to_layout_tensor[is_async=is_async](
            dst_thread_local, src_thread_local, src_element_layout
        )


# Copies an nd-buffer fragment to `thread_id` thread local LayoutTensor element
# where each element of the fragment is originally distributed by `thread_layout`.
#
@always_inline("nodebug")
fn copy_from_nd_buffer_masked[
    src_rank: Int,
    dtype: DType,
    dst_data_layout: Layout,
    src_buff_shape: DimList,
    thread_layout: Layout,
    is_async: Bool = False,
    swizzle: OptionalReg[_swizzle_signature] = None,
](
    dst_thread_local: LayoutTensor[
        mut=True,
        dtype,
        dst_data_layout,
        *_, **_,
    ],
    src: NDBuffer[dtype, src_rank, src_buff_shape],
    tile_mask: TileMask,
    thread_id: Int,
):
    alias dst_rank = dst_data_layout.rank()
    alias dst_element_layout = dst_thread_local.element_layout
    # FIXME: Relax this to support any ranked data and thread layouts.
    constrained[src_rank == 2, "Only rank-2 layouts is supported for now."]()

    constrained[src_rank == dst_rank, "src and dst should have same rank"]()

    constrained[
        dst_thread_local.element_layout.rank() == 1
        or dst_thread_local.element_layout.rank() == 2,
        "Only rank-1, rank-2 vectoriztion is supported",
    ]()

    alias threads_layout_rank = thread_layout.rank()
    constrained[
        threads_layout_rank == dst_rank,
        "thread and data layout should have the same rank",
    ]()

    @parameter
    if dst_element_layout.rank() == 1:
        var src_vectorized = vectorize[1, Int(dst_element_layout.shape[0])](src)
        var vec_mask = _vectorize_mask[
            sizes = (1, Int(dst_element_layout.shape[0]))
        ](tile_mask)
        var src_vectorized_buffer = src_vectorized[0]
        var src_element_layout = src_vectorized[1]
        alias element_size = Int(dst_element_layout.shape[0])
        var src_thread_local = distribute[
            thread_layout=thread_layout,
            swizzle=swizzle,
            element_size=element_size,
        ](src_vectorized_buffer, thread_id)
        var distribute_mask = _distribute_mask[thread_layout=thread_layout](
            vec_mask, thread_id
        )

        _copy_nd_buffer_to_layout_tensor_masked[is_async=is_async](
            dst_thread_local,
            src_thread_local,
            src_element_layout,
            distribute_mask,
        )
    elif dst_element_layout.rank() == 2:
        var src_vectorized = vectorize[
            Int(dst_element_layout.shape[0]),
            Int(dst_element_layout.shape[1]),
        ](src)
        var vec_mask = _vectorize_mask[
            sizes = (
                Int(dst_element_layout.shape[0]),
                Int(dst_element_layout.shape[1]),
            )
        ](tile_mask)
        var src_vectorized_buffer = src_vectorized[0]
        var src_element_layout = src_vectorized[1]
        alias element_size = Int(dst_element_layout.shape[1])
        var src_thread_local = distribute[
            thread_layout=thread_layout,
            swizzle=swizzle,
            element_size=element_size,
        ](src_vectorized_buffer, thread_id)
        var distribute_mask = _distribute_mask[thread_layout=thread_layout](
            vec_mask, thread_id
        )
        _copy_nd_buffer_to_layout_tensor_masked[is_async=is_async](
            dst_thread_local,
            src_thread_local,
            src_element_layout,
            distribute_mask,
        )


# Copies LayoutTensor element into `dst` NDBuffer fragments, where each element
# of the fragment is distributed by `thread_layout`.
#
@always_inline("nodebug")
fn copy_to_nd_buffer[
    dst_rank: Int,
    dtype: DType,
    dst_buff_shape: DimList,
    src_data_layout: Layout,
    thread_layout: Layout,
](
    dst: NDBuffer[dtype, dst_rank, dst_buff_shape],
    src_thread_local: LayoutTensor[
        dtype,
        src_data_layout,
        *_, **_,
    ],
    thread_id: Int,
):
    alias src_rank = src_data_layout.rank()
    alias src_element_layout = src_thread_local.element_layout
    # FIXME: Relax this to support any ranked data and thread layouts.
    constrained[src_rank == 2, "Only rank-2 layouts is supported for now."]()

    constrained[src_rank == dst_rank, "src and dst should have same rank"]()

    constrained[
        src_thread_local.element_layout.rank() == 1
        or src_thread_local.element_layout.rank() == 2,
        "Only rank-1, rank-2 vectoriztion is supported",
    ]()

    alias threads_layout_rank = thread_layout.rank()
    constrained[
        threads_layout_rank == dst_rank,
        "thread and data layout should have the same rank",
    ]()

    @parameter
    if src_element_layout.rank() == 1:
        var dst_vectorized = vectorize[1, Int(src_element_layout.shape[0])](dst)
        var dst_vectorized_buffer = dst_vectorized[0]
        var dst_element_layout = dst_vectorized[1]
        var dst_thread_local = distribute[thread_layout=thread_layout](
            dst_vectorized_buffer, thread_id
        )
        _copy_layout_tensor_to_nd_buffer(
            dst_thread_local, dst_element_layout, src_thread_local
        )
    else:
        var dst_vectorized = vectorize[
            Int(src_element_layout.shape[0]),
            Int(src_element_layout.shape[1]),
        ](dst)
        var dst_vectorized_buffer = dst_vectorized[0]
        var dst_element_layout = dst_vectorized[1]
        var dst_thread_local = distribute[thread_layout=thread_layout](
            dst_vectorized_buffer, thread_id
        )
        _copy_layout_tensor_to_nd_buffer(
            dst_thread_local, dst_element_layout, src_thread_local
        )


@always_inline("nodebug")
fn copy_to_nd_buffer_masked[
    dst_rank: Int,
    dtype: DType,
    dst_buff_shape: DimList,
    src_data_layout: Layout,
    thread_layout: Layout,
](
    dst: NDBuffer[dtype, dst_rank, dst_buff_shape],
    src_thread_local: LayoutTensor[
        dtype,
        src_data_layout,
        *_, **_,
    ],
    tile_mask: TileMask,
    thread_id: Int,
):
    alias src_rank = src_data_layout.rank()
    alias src_element_layout = src_thread_local.element_layout
    # FIXME: Relax this to support any ranked data and thread layouts.
    constrained[src_rank == 2, "Only rank-2 layouts is supported for now."]()

    constrained[src_rank == dst_rank, "src and dst should have same rank"]()

    constrained[
        src_thread_local.element_layout.rank() == 1
        or src_thread_local.element_layout.rank() == 2,
        "Only rank-1, rank-2 vectoriztion is supported",
    ]()

    alias threads_layout_rank = thread_layout.rank()
    constrained[
        threads_layout_rank == dst_rank,
        "thread and data layout should have the same rank",
    ]()

    @parameter
    if src_element_layout.rank() == 1:
        var dst_vectorized = vectorize[1, Int(src_element_layout.shape[0])](dst)
        var vectorize_mask = _vectorize_mask[
            sizes = (1, Int(src_element_layout.shape[0]))
        ](tile_mask)
        var dst_vectorized_buffer = dst_vectorized[0]
        var dst_element_layout = dst_vectorized[1]
        var dst_thread_local = distribute[thread_layout=thread_layout](
            dst_vectorized_buffer, thread_id
        )
        var distribute_mask = _distribute_mask[thread_layout](
            vectorize_mask, thread_id
        )
        _copy_layout_tensor_to_nd_buffer_masked(
            dst_thread_local,
            dst_element_layout,
            src_thread_local,
            distribute_mask,
        )
    else:
        var dst_vectorized = vectorize[
            Int(src_element_layout.shape[0]),
            Int(src_element_layout.shape[1]),
        ](dst)
        var vectorize_mask = _vectorize_mask[
            sizes = (
                Int(src_element_layout.shape[0]),
                Int(src_element_layout.shape[1]),
            )
        ](tile_mask)
        var dst_vectorized_buffer = dst_vectorized[0]
        var dst_element_layout = dst_vectorized[1]
        var dst_thread_local = distribute[thread_layout=thread_layout](
            dst_vectorized_buffer, thread_id
        )
        var distribute_mask = _distribute_mask[thread_layout](
            vectorize_mask, thread_id
        )
        _copy_layout_tensor_to_nd_buffer_masked(
            dst_thread_local,
            dst_element_layout,
            src_thread_local,
            distribute_mask,
        )


# Copies `src_buffer` to `dst_tensor` asynchronously, the work is distributed
# to `thread_layout` of threads.
#
fn copy_from_nd_buffer_async[
    src_rank: Int,
    dtype: DType,
    dst_data_layout: Layout,
    src_buff_shape: DimList,
    thread_layout: Layout,
    is_async: Bool = False,
    swizzle: OptionalReg[_swizzle_signature] = None,
](
    dst_tensor: LayoutTensor[
        mut=True,
        dtype,
        dst_data_layout,
        *_, **_,
    ],
    src_buffer: NDBuffer[dtype, src_rank, src_buff_shape],
):
    copy_from_nd_buffer[thread_layout=thread_layout, is_async=True](
        dst_tensor.distribute[thread_layout](thread_idx.x),
        src_buffer,
        Int(thread_idx.x),
    )


fn from_ndbuffer_row_major(
    buffer: NDBuffer,
    out result: LayoutTensor[
        mut=True,
        buffer.type,
        Layout.row_major[buffer.rank](buffer.shape),
        buffer.origin,
        address_space = buffer.address_space,
    ],
):
    """This function takes the underlying buffer from NDBuffer without explicitly
    copying any data.
    """
    var runtime_layout = __type_of(result.runtime_layout).row_major[
        buffer.rank
    ](buffer.get_shape())
    return __type_of(result)(
        buffer.data,
        runtime_layout,
    )
