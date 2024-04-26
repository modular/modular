# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from layout import LayoutTensor, Layout
from layout.int_tuple import to_int, flatten

from buffer import NDBuffer
from buffer.list import (
    DimList,
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
    thread_layout: Layout,
](
    dst_thread_local: LayoutTensor[
        dtype, dst_data_layout, dst_rank, address_space=dst_address_space
    ],
    src: NDBuffer[dtype, src_rank, src_buff_shape],
    thread_id: Int,
):
    # FIXME: Relax this to support any ranked data and thread layouts.
    constrained[src_rank == 2, "Only rank-2 layouts is supported for now."]()

    constrained[src_rank == dst_rank, "src and dst should have same rank"]()

    constrained[
        dst_thread_local.element_layout == Layout(1, 1),
        "Scalar element layout is only supported for now",
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

            @parameter
            fn compute_offset[i: Int]():
                var fragments_stride_i = src_tile_i_j.dynamic_stride[i]
                alias shape_i = to_int(flatten(thread_layout.shape)[i])
                alias stride_i = to_int(flatten(thread_layout.stride)[i])
                var thread_coord_i = (thread_id // stride_i) % shape_i
                offset += thread_coord_i * fragments_stride_i

            unroll[compute_offset, threads_layout_rank]()

            var src_val = src_tile_i_j.data.offset(offset).load()
            dst_thread_local[th_0, th_1] = rebind[
                dst_thread_local.element_type
            ](src_val)
