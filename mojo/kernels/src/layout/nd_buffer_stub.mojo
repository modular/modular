# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from layout import LayoutTensor, Layout
from layout.int_tuple import to_int, flatten, depth
from layout.layout import make_layout

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


# Returns the linear index of an element, this is equivalent to concat
# the element layout and the buffer layout
fn _get_element_idx[
    rank: Int,
    dtype: DType,
    shape: DimList,
    element_shape: StaticIntTuple[rank],
](
    linear_coord: Int,
    buff: NDBuffer[dtype, rank, shape],
    element_layout: ElementLayout[rank, element_shape],
) -> Int:
    var result = 0
    var curr_linear_crd = linear_coord

    # evaluate according to
    # iterate over outer most
    @unroll
    for i in range(rank):
        result += (
            curr_linear_crd % element_layout.shape[i]
        ) * element_layout.stride[i]
        curr_linear_crd = curr_linear_crd // element_layout.shape[i]

    @unroll
    for i in range(rank):
        result += (
            curr_linear_crd % buff.dynamic_shape[i]
        ) * buff.dynamic_stride[i]
        curr_linear_crd = curr_linear_crd // buff.dynamic_shape[i]

    return result


fn _get_element_idx[
    rank: Int,
    dtype: DType,
    shape: DimList,
](linear_coord: Int, buff: NDBuffer[dtype, rank, shape]) -> Int:
    var result = 0
    var curr_linear_crd = linear_coord

    @unroll
    for i in range(rank):
        result += (
            curr_linear_crd % buff.dynamic_shape[i]
        ) * buff.dynamic_stride[i]
        curr_linear_crd = curr_linear_crd // buff.dynamic_shape[i]
    return result


fn _get_element_idx[
    rank: Int,
    element_shape: StaticIntTuple[rank],
](
    linear_coord: Int,
    element_layout: ElementLayout[rank, element_shape],
) -> Int:
    var result = 0
    var curr_linear_crd = linear_coord

    # evaluate according to
    # iterate over outer most
    @unroll
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


fn _copy_nd_buffer_to_layout_tensor[
    dst_rank: Int,
    src_rank: Int,
    dtype: DType,
    layout: Layout,
    shape: DimList,
    dst_address_space: AddressSpace,
    tensor_element_layout: Layout,
    buff_element_layout_shape: StaticIntTuple[src_rank],
](
    dst: LayoutTensor[
        dtype,
        layout,
        dst_rank,
        address_space=dst_address_space,
        element_layout=tensor_element_layout,
    ],
    src: NDBuffer[dtype, src_rank, shape],
    buff_element_layout: ElementLayout[src_rank, buff_element_layout_shape],
):
    alias num_elements = dst.layout.size()
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

        alias vec_size = to_int(tensor_element_layout.shape[0])
        alias alignment = alignof[dst.element_type]()

        @parameter
        fn _copy_vector[i: Int]():
            alias dst_idx = make_layout(tensor_element_layout, dst.layout)(
                i * vec_size
            )
            var src_idx = _get_element_idx(
                i * vec_size, src, buff_element_layout
            )

            var src_element = src.data.offset(src_idx).load[
                width=vec_size, alignment=alignment
            ]()
            dst.ptr.offset(dst_idx).store[width=vec_size, alignment=alignment](
                src_element
            )

        unroll[_copy_vector, num_elements]()
    # 2d-vector load/store
    elif (
        tensor_element_layout.rank() == 2
        and tensor_element_layout.stride[1] == 1
    ):
        alias num_copies = tensor_element_layout.shape[0].value()
        alias vec_width = tensor_element_layout.shape[1].value()

        @parameter
        fn copy_by_element[i: Int]():
            alias dst_offset = layout(i)
            var src_offset = _get_element_idx(i, src)

            @parameter
            fn copy_by_vec[j: Int]():
                alias dst_idx = dst_offset + tensor_element_layout(j)
                var src_idx = src_offset + _get_element_idx(
                    j, buff_element_layout
                )

                var src_vec = src.data.load[
                    width=vec_width,
                    alignment = alignof[SIMD[dtype, vec_width]](),
                ](src_idx).cast[dtype]()

                dst.ptr.store[
                    width=vec_width,
                    alignment = alignof[SIMD[dtype, vec_width]](),
                ](dst_idx, src_vec)

            unroll[copy_by_vec, num_copies]()

        unroll[copy_by_element, num_elements]()
    # Scalar case.
    else:

        @parameter
        fn _copy_element[i: Int]():
            alias dst_idx = make_layout(tensor_element_layout, dst.layout)(i)
            var src_idx = _get_element_idx(i, src, buff_element_layout)
            dst.ptr[dst_idx] = src.data[src_idx]

        unroll[_copy_element, num_elements * dst.element_size]()


fn _copy_layout_tensor_to_nd_buffer[
    dst_rank: Int,
    src_rank: Int,
    dtype: DType,
    layout: Layout,
    shape: DimList,
    dst_address_space: AddressSpace,
    tensor_element_layout: Layout,
    buff_element_layout_shape: StaticIntTuple[dst_rank],
](
    dst: NDBuffer[dtype, dst_rank, shape],
    buff_element_layout: ElementLayout[dst_rank, buff_element_layout_shape],
    src: LayoutTensor[
        dtype,
        layout,
        src_rank,
        address_space=dst_address_space,
        element_layout=tensor_element_layout,
    ],
):
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

        alias vec_size = to_int(tensor_element_layout.shape[0])
        alias alignment = alignof[src.element_type]()

        @parameter
        fn _copy_vector[i: Int]():
            var dst_idx = _get_element_idx(
                i * vec_size, dst, buff_element_layout
            )
            alias src_idx = make_layout(tensor_element_layout, src.layout)(
                i * vec_size
            )

            var src_element = src.ptr.offset(src_idx).load[
                width=vec_size, alignment=alignment
            ]()
            dst.data.offset(dst_idx).store[width=vec_size, alignment=alignment](
                src_element
            )

        unroll[_copy_vector, num_elements]()
    # 2d-vector load/store
    elif (
        tensor_element_layout.rank() == 2
        and tensor_element_layout.stride[1] == 1
    ):
        alias num_copies = tensor_element_layout.shape[0].value()
        alias vec_width = tensor_element_layout.shape[1].value()

        @parameter
        fn copy_by_element[i: Int]():
            # Offset to the current element.
            var dst_offset = _get_element_idx(i, dst)

            alias src_offset = layout(i)

            @parameter
            fn copy_by_vec[j: Int]():
                var dst_idx = dst_offset + _get_element_idx(
                    j, buff_element_layout
                )
                alias src_idx = src_offset + tensor_element_layout(j)

                var src_vec = src.ptr.load[
                    width=vec_width,
                    alignment = alignof[SIMD[dtype, vec_width]](),
                ](src_idx).cast[dtype]()

                dst.data.store[
                    width=vec_width,
                    alignment = alignof[SIMD[dtype, vec_width]](),
                ](dst_idx, src_vec)

            unroll[copy_by_vec, num_copies]()

        unroll[copy_by_element, num_elements]()
    # Scalar case.
    else:

        @parameter
        fn _copy_element[i: Int]():
            alias src_idx = make_layout(tensor_element_layout, src.layout)(i)
            var dst_idx = _get_element_idx(i, dst, buff_element_layout)
            dst.data[dst_idx] = src.ptr[src_idx]

        unroll[_copy_element, num_elements * src.element_size]()


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
