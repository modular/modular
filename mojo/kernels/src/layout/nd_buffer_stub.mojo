# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from layout import LayoutTensor, Layout
from layout.int_tuple import to_int, flatten, depth

from buffer import NDBuffer
from buffer.list import DimList, Dim


# Returns the shape of distribute `thread_layout` into `shape`.
#
fn __distribute_shape[thread_layout: Layout](shape: DimList) -> DimList:
    constrained[
        thread_layout.rank() <= 3,
        "__distribute_shape requires thread_layout <= 3",
    ]()

    var res = StaticTuple[Dim][thread_layout.rank()]()

    @parameter
    fn _get_dim[i: Int]():
        if shape.at[i]().is_dynamic():
            res[i] = Dim()
        else:
            res[i] = shape.at[i]() // to_int(thread_layout.shape[i])

    unroll[_get_dim, thread_layout.rank()]()

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
    _result_shape: DimList = __distribute_shape[thread_layout](shape),
](buff: NDBuffer[dtype, rank, shape], thread_id: Int) -> NDBuffer[
    dtype, rank, _result_shape
]:
    constrained[
        depth(thread_layout.shape) == 1,
        "distribute threads to NDBuffer only supports depth-1 thread layouts",
    ]()

    var res_strides = StaticIntTuple[rank]()
    var res_shape = StaticIntTuple[rank]()

    @parameter
    fn _fill_shape_and_stride[i: Int]():
        alias thread_shape_i = to_int(thread_layout.shape[i])
        res_shape[i] = buff.dynamic_shape[i] // thread_shape_i
        res_strides[i] = buff.dynamic_stride[i] * thread_shape_i

    unroll[_fill_shape_and_stride, rank]()

    var thread_offset = 0

    @parameter
    fn _compute_offset[i: Int]():
        alias shape_i = to_int(thread_layout.shape[i])
        alias stride_i = to_int(thread_layout.stride[i])
        var thread_coords_i = (thread_id // stride_i) % shape_i
        thread_offset += thread_coords_i * buff.dynamic_stride[i]

    unroll[_compute_offset, rank]()

    var res = NDBuffer[dtype, rank, _result_shape](
        buff.data.offset(thread_offset),
        dynamic_shape=res_shape,
        dynamic_stride=res_strides,
    )
    return res


# FIXME: Move to a shared utility.
# Returns the size of variadic integer parameters.
#
fn __get_len[*var_int: Int]() -> Int:
    return __mlir_op.`pop.variadic.size`(var_int)


fn __vectorize_shape[*sizes: Int](shape: DimList) -> DimList:
    alias rank = __get_len[sizes]()

    constrained[
        rank <= 3,
        "__vectorize_shape vector sizes <= 3",
    ]()

    var res = StaticTuple[Dim, rank]()

    @parameter
    fn _fill_shape[i: Int]():
        alias size_i = sizes[i]

        if shape.at[i]().is_dynamic():
            res[i] = Dim()
        else:
            res[i] = shape.at[i]() // size_i

    unroll[_fill_shape, rank]()

    @parameter
    if rank == 1:
        return DimList(res[0])
    elif rank == 2:
        return DimList(res[0], res[1])
    elif rank == 3:
        return DimList(res[0], res[1], res[2])
    return DimList()


fn __to_static_tuple[*sizes: Int, rank: Int]() -> StaticIntTuple[rank]:
    var vals = StaticIntTuple[rank]()

    @parameter
    fn _fill[i: Int]():
        vals[i] = sizes[i]

    unroll[_fill, rank]()
    return vals


# Stores the layout of the vectorized buffer element.
#
struct ElementLayout[rank: Int, shape: StaticIntTuple[rank]](
    CollectionElement, Stringable
):
    var stride: StaticIntTuple[rank]

    fn __init__(inout self):
        self.stride = StaticIntTuple[rank]()

    fn __copyinit__(inout self, exisiting: Self):
        self.stride = exisiting.stride

    fn __moveinit__(inout self, owned exisiting: Self):
        self.stride = exisiting.stride

    fn __str__(self) -> String:
        return shape.__str__() + ":" + self.stride


# Vectorizes buffer and returns the vecrtorized buffer and its dynamic layout.
#
@always_inline("nodebug")
fn vectorize[
    *sizes: Int,
    dtype: DType,
    rank: Int,
    shape: DimList,
    _res_shape: DimList = __vectorize_shape[sizes](shape),
](
    buff: NDBuffer[
        dtype,
        rank,
        shape,
    ]
) -> Tuple[
    NDBuffer[dtype, rank, _res_shape],
    ElementLayout[rank, __to_static_tuple[sizes, rank=rank]()],
]:
    var buff_shape = StaticIntTuple[rank]()
    var buff_stride = StaticIntTuple[rank]()

    var element_layout = ElementLayout[
        rank, __to_static_tuple[sizes, rank=rank]()
    ]()

    @parameter
    fn _fill_layout_data[i: Int]():
        element_layout.stride[i] = buff.dynamic_stride[i]
        buff_shape[i] = buff.dynamic_shape[i] // sizes[i]
        buff_stride[i] = buff.dynamic_stride[i] * sizes[i]

    unroll[_fill_layout_data, rank]()

    return Tuple(
        NDBuffer[dtype, rank, _res_shape](
            buff.data, dynamic_shape=buff_shape, dynamic_stride=buff_stride
        ),
        element_layout,
    )


# Copies an nd-buffer fragment to `thread_id` thread local LayoutTensor element
# where each element of the fragment is originally distributed by `thread_layout`.
#
@always_inline("nodebug")
fn copy_from_nd_buffer[
    dst_rank: Int,
    src_rank: Int,
    dtype: DType,
    dst_address_space: AddressSpace,
    dst_data_layout: Layout,
    src_buff_shape: DimList,
    dst_element_layout: Layout,
    thread_layout: Layout,
](
    dst_thread_local: LayoutTensor[
        dtype,
        dst_data_layout,
        dst_rank,
        address_space=dst_address_space,
        element_layout=dst_element_layout,
    ],
    src: NDBuffer[dtype, src_rank, src_buff_shape],
    thread_id: Int,
):
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

    alias dim_0_shape = to_int(dst_thread_local.shape[0]())
    alias dim_1_shape = to_int(dst_thread_local.shape[1]())

    alias tile_dim_0 = to_int(thread_layout.shape[0])
    alias tile_dim_1 = to_int(thread_layout.shape[1])

    for th_0 in range(dim_0_shape):
        for th_1 in range(dim_1_shape):
            # TODO: Can we do tile and then reuse it instead of evaluating it every time ?
            var src_tile_i_j = src.tile[tile_dim_0, tile_dim_1](
                tile_coords=(th_0, th_1)
            )

            var offset = 0

            # Returns the coords of the current thread.
            @parameter
            fn get_thread_coords() -> StaticIntTuple[src_rank]:
                var thread_coords = StaticIntTuple[src_rank]()

                @parameter
                fn fill[i: Int]():
                    alias shape_i = to_int(flatten(thread_layout.shape)[i])
                    alias stride_i = to_int(flatten(thread_layout.stride)[i])
                    thread_coords[i] = (thread_id // stride_i) % shape_i

                unroll[fill, src_rank]()
                return thread_coords

            var thread_coords = get_thread_coords()

            # Returns the stride of vectorized elements of the current thread.
            @parameter
            fn get_thread_vec_stride() -> StaticIntTuple[src_rank]:
                var strides = StaticIntTuple[src_rank](1)

                @parameter
                if dst_thread_local.element_layout.rank() == 1:
                    strides[1] = to_int(
                        dst_thread_local.element_layout.shape[0]
                    )

                @parameter
                if dst_thread_local.element_layout.rank() == 2:
                    strides[0] = to_int(
                        dst_thread_local.element_layout.shape[0]
                    )
                    strides[1] = to_int(
                        dst_thread_local.element_layout.shape[1]
                    )
                return strides

            var thread_vec_stride = get_thread_vec_stride()

            # Compute the offset of the element taking into account the
            @parameter
            fn compute_offset[i: Int]():
                var fragments_stride_i = src_tile_i_j.dynamic_stride[i]
                offset += (
                    thread_coords[i] * thread_vec_stride[i] * fragments_stride_i
                )

            unroll[compute_offset, threads_layout_rank]()

            alias vec_dim: Int = 0 if dst_thread_local.element_layout.rank() == 1 else 1
            alias vec_shape = to_int(
                dst_thread_local.element_layout.shape[vec_dim]
            )

            @parameter
            if dst_thread_local.element_layout.rank() == 1:
                var src_val = src_tile_i_j.data.offset(offset).load[
                    width=vec_shape
                ]()
                dst_thread_local[th_0, th_1] = rebind[
                    dst_thread_local.element_type
                ](src_val)
            else:
                alias num_vec = to_int(dst_thread_local.element_layout.shape[0])
                var res_vec = SIMD[dtype, dst_thread_local.element_size]()

                @unroll
                for i in range(num_vec):
                    alias eleme_stride = to_int(
                        dst_thread_local.element_layout.stride[0]
                    )
                    var vec_offset = offset + i * src_tile_i_j.dynamic_stride[0]
                    var src_val = src_tile_i_j.data.offset(vec_offset).load[
                        width=vec_shape
                    ]()

                    #  2D rank vectors are stored columnwise.
                    @unroll
                    for j in range(vec_shape):
                        res_vec[j * vec_shape + i] = src_val[j]

                dst_thread_local[th_0, th_1] = rebind[
                    dst_thread_local.element_type
                ](res_vec)


# Copies LayoutTensor element into `dst` NDBuffer fragments, where each element
# of the fragment is distributed by `thread_layout`.
#
@always_inline("nodebug")
fn copy_to_nd_buffer[
    dst_rank: Int,
    src_rank: Int,
    dtype: DType,
    dst_buff_shape: DimList,
    dst_address_space: AddressSpace,
    src_data_layout: Layout,
    src_element_layout: Layout,
    thread_layout: Layout,
](
    dst: NDBuffer[dtype, dst_rank, dst_buff_shape],
    src_thread_local: LayoutTensor[
        dtype,
        src_data_layout,
        src_rank,
        address_space=dst_address_space,
        element_layout=src_element_layout,
    ],
    thread_id: Int,
):
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

    alias dim_0_shape = to_int(src_thread_local.shape[0]())
    alias dim_1_shape = to_int(src_thread_local.shape[1]())

    alias tile_dim_0 = to_int(thread_layout.shape[0])
    alias tile_dim_1 = to_int(thread_layout.shape[1])

    for th_0 in range(dim_0_shape):
        for th_1 in range(dim_1_shape):
            # TODO: Can we do tile and then reuse it instead of evaluating it every time ?
            var src_tile_i_j = dst.tile[tile_dim_0, tile_dim_1](
                tile_coords=(th_0, th_1)
            )

            var offset = 0

            # Returns the coords of the current thread.
            @parameter
            fn get_thread_coords() -> StaticIntTuple[src_rank]:
                var thread_coords = StaticIntTuple[src_rank]()

                @parameter
                fn fill[i: Int]():
                    alias shape_i = to_int(flatten(thread_layout.shape)[i])
                    alias stride_i = to_int(flatten(thread_layout.stride)[i])
                    thread_coords[i] = (thread_id // stride_i) % shape_i

                unroll[fill, src_rank]()
                return thread_coords

            var thread_coords = get_thread_coords()

            # Returns the stride of vectorized elements of the current thread.
            @parameter
            fn get_thread_vec_stride() -> StaticIntTuple[src_rank]:
                var strides = StaticIntTuple[src_rank](1)

                @parameter
                if src_thread_local.element_layout.rank() == 1:
                    strides[1] = to_int(
                        src_thread_local.element_layout.shape[0]
                    )

                @parameter
                if src_thread_local.element_layout.rank() == 2:
                    strides[0] = to_int(
                        src_thread_local.element_layout.shape[0]
                    )
                    strides[1] = to_int(
                        src_thread_local.element_layout.shape[1]
                    )
                return strides

            var thread_vec_stride = get_thread_vec_stride()

            # Compute the offset of the element taking into account the
            @parameter
            fn compute_offset[i: Int]():
                var fragments_stride_i = src_tile_i_j.dynamic_stride[i]
                offset += (
                    thread_coords[i] * thread_vec_stride[i] * fragments_stride_i
                )

            unroll[compute_offset, threads_layout_rank]()

            alias vec_dim: Int = 0 if src_thread_local.element_layout.rank() == 1 else 1
            alias vec_shape = to_int(
                src_thread_local.element_layout.shape[vec_dim]
            )

            @parameter
            if src_thread_local.element_layout.rank() == 1:
                src_tile_i_j.data.offset(offset).store[
                    width = src_thread_local.element_size
                ](src_thread_local[th_0, th_1])
            else:
                alias num_vecs = to_int(
                    src_thread_local.element_layout.shape[0]
                )

                @parameter
                fn store_slice[i: Int]():
                    var vec_offset = offset + i * src_tile_i_j.dynamic_stride[0]
                    src_tile_i_j.data.offset(vec_offset).store[width=vec_shape](
                        src_thread_local[th_0, th_1].slice[
                            vec_shape, offset = i * vec_shape
                        ]()
                    )

                unroll[store_slice, num_vecs]()
