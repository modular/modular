# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param
from Buffer import NDBuffer, prod_dims
from DType import DType
from Functional import (
    vectorize,
    vectorize_unroll,
    async_parallelize,
    unroll,
    elementwise,
)
from Index import StaticIntTuple
from Intrinsics import PrefetchOptions
from LLCL import OutputChainPtr
from List import DimList
from Math import div_ceil
from Math import min
from Range import range
from SIMD import SIMD
from TargetInfo import dtype_sizeof
from TypeUtilities import rebind

## gather_reduce_2D_axis_1
@adaptive
fn gather_reduce[
    output_rank: Int,
    output_shape: DimList,
    input_rank: Int,
    input_shape: DimList,
    indices_rank: Int,
    indices_shape: DimList,
    type: DType,
    gather_axis: Int,
    reduce_axis: Int,
    simd_width: Int,
    reduce_fn: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width]
    ) -> SIMD[type, width],
](
    output: NDBuffer[output_rank, output_shape, type],
    input: NDBuffer[input_rank, input_shape, type],
    indices: NDBuffer[
        indices_rank,
        indices_shape,
        DType.int32,
    ],
    reduce_init: SIMD[type, 1],
    out_chain: OutputChainPtr,
):
    """Computes output[i, j, k] = input[indices[i, j], k] and simultaneously
    reduces the output accross axis 1 to produce output[i, k].

    The motivating use-case for this is multi-hot embeddings in recommender models.
    This provides similar functionality to Torch's EmbeddingBag layer. In that
    context, i is the batch dimension, j is the multi-hot dimension, and k is
    the embedding dimension.
    """
    assert_param[input_rank == 2]()
    assert_param[indices_rank == 2]()
    assert_param[gather_axis == 0]()
    assert_param[reduce_axis == 1]()

    # Short-circuit for trivial cases, and to avoid divide-by-zero
    if input.size() == 0 or indices.size() == 0:
        return

    # TODO: find a heuristic to replace the magic number.
    # This is about 4x larger than the default in gather, which makes sense
    # since this kernel performs far fewer writes
    alias MIN_TASK_COPY_SIZE = 64 * 100 * 32 * 4  # bytes
    let num_threads = out_chain.get_runtime().parallelism_level()
    let num_tasks = min(
        div_ceil(
            indices.dim[0]()
            * indices.dim[1]()
            * input.dim[1]()
            * dtype_sizeof[type](),
            MIN_TASK_COPY_SIZE,
        ),
        num_threads,
    )

    let num_chunks_per_task = div_ceil(indices.dim[0](), num_tasks)

    var output_2d_dims = StaticIntTuple[2](output.dim[0](), output.dim[1]())

    @parameter
    if output_rank == 3:
        output_2d_dims[1] = output.dim[2]()

    let output_bind = NDBuffer[2, DimList.create_unknown[2](), type](
        output.data, output_2d_dims, type
    )
    let input_bind = rebind[NDBuffer[2, DimList.create_unknown[2](), type]](
        input
    )
    let indices_bind = rebind[
        NDBuffer[indices_rank, indices_shape, DType.int32]
    ](indices)

    @always_inline
    @parameter
    fn task_func(task_id: Int):
        alias unroll_factor = 2
        alias prefetch_offset = -1
        alias usimd_width = unroll_factor * simd_width

        let output = output_bind
        let input = input_bind
        let indices = indices_bind
        let row_size = output.dim[1]()

        # need to reduce on an entire 2D slice at a time, otherwise multiple
        # threads will try to accumulate in the same buffer simultaneously
        let start_slice = task_id * num_chunks_per_task
        let end_slice = min(
            (task_id + 1) * num_chunks_per_task, indices.dim[0]()
        )

        for i in range(start_slice, end_slice):

            @always_inline
            @parameter
            fn _accum_in_place[simd_width: Int](k: Int):
                var accum = SIMD[type, simd_width](reduce_init)
                for j in range(indices.dim[1]()):
                    """Computes output[i,k] = reduction over j of (input[indices[i,j],k])
                    for j in range [0,indices.dim[1])"""
                    let idx = indices[i, j].value

                    # min so that we don't read beyond end of indices
                    @parameter
                    if prefetch_offset > 0:
                        let clamped_prefetch_offset = min(
                            prefetch_offset,
                            indices.dim[0]() * indices.dim[1]()
                            - (i * indices.dim[1]() + j)
                            - 1,
                        )
                        let next_idx_ptr = indices._offset(
                            StaticIntTuple[indices_rank](i, j)
                        ) + clamped_prefetch_offset
                        input.prefetch[
                            PrefetchOptions()
                            .for_read()
                            .high_locality()
                            .to_data_cache()
                        ](next_idx_ptr.load().to_int(), 0)

                    let in_idx = StaticIntTuple[2](idx, k)

                    let gather_chunk = input.simd_load[simd_width](in_idx)
                    accum = reduce_fn[type, simd_width](accum, gather_chunk)

                let out_idx = StaticIntTuple[2](i, k)
                output.simd_store[simd_width](out_idx, accum)

            vectorize[usimd_width, _accum_in_place](row_size)

    async_parallelize[task_func](out_chain, num_tasks)


@adaptive
fn gather[
    output_rank: Int,
    output_shape: DimList,
    input_rank: Int,
    input_shape: DimList,
    indices_rank: Int,
    indices_shape: DimList,
    type: DType,
    indices_type: DType,
    axis: Int,
    simd_width: Int,
](
    output: NDBuffer[output_rank, output_shape, type],
    input: NDBuffer[input_rank, input_shape, type],
    indices: NDBuffer[indices_rank, indices_shape, indices_type],
    out_chain: OutputChainPtr,
):
    """Gather operation as defined in https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather.

    Note that this is NOT the same as the default PyTorch gather (which is equivalent to
    https://github.com/onnx/onnx/blob/main/docs/Operators.md#gatherelements).
    """

    # Gathers contiguous rows of the input
    assert_param[axis != input_rank - 1]()

    let indices_len = indices.size()
    # Short-circuit for trivial cases, and to avoid divide-by-zero
    if input.size() == 0 or indices_len == 0:
        # No-op
        out_chain.mark_ready()
        return

    alias prefetch_offset = 12  # TODO: search

    let end_indices_ptr = indices.flatten().data.offset(indices.size())

    @always_inline
    @parameter
    fn gather_fn[width: Int, rank: Int](out_coords: StaticIntTuple[rank]):
        # out_coords consists of 3 chunks:
        #   out_coords[0:axis] = input coords[0:axis]
        #   out_coords[axis:axis+indices_rank] = indices_coords
        #   out_coords[axis + indices_rank:] = input_coords[axis + 1:]
        # and input_coords[axis] = indices[indices_coords]
        var indices_coords = StaticIntTuple[indices_rank]()
        var input_coords = StaticIntTuple[input_rank]()

        @always_inline
        @parameter
        fn _get_indices_coords[idx: Int]():
            indices_coords[idx] = out_coords[axis + idx]

        unroll[indices_rank, _get_indices_coords]()

        let idx = indices[indices_coords]

        @always_inline
        @parameter
        fn _get_input_coords_before_ax[idx: Int]():
            input_coords[idx] = out_coords[idx]

        unroll[axis, _get_input_coords_before_ax]()

        @always_inline
        @parameter
        fn _get_input_coords_after_ax[idx: Int]():
            input_coords[idx + axis + 1] = out_coords[idx + axis + indices_rank]

        unroll[output_rank - axis - 1, _get_input_coords_after_ax]()

        @parameter
        if prefetch_offset > 0:
            let indices_ptr = indices._offset(indices_coords)
            let indices_remaining = (
                end_indices_ptr.__as_index() - indices_ptr.__as_index()
            ) // dtype_sizeof[type]()
            # assumes that indices are layed out in row major order
            let next_idx_ptr = indices._offset(indices_coords) + min(
                indices_remaining, prefetch_offset
            )
            input_coords[axis] = next_idx_ptr.load().to_int()
            input.prefetch[
                PrefetchOptions().for_read().high_locality().to_data_cache()
            ](input_coords)

        input_coords[axis] = idx.to_int()

        let input_val = input.simd_load[width](input_coords)
        output.simd_store[width](
            rebind[StaticIntTuple[output_rank]](out_coords), input_val
        )

    alias unroll_factor = 1
    # Elementwise generator calls gather_fn on all coords in output.
    # We can determine the input element to gather using the information
    # in the output coords only.
    # This also enables vectorization since we are gather rows of the input.
    elementwise[output_rank, simd_width, unroll_factor, gather_fn](
        output.get_shape(), out_chain
    )


# gather_ND_input_MD_indices
@adaptive
fn gather[
    output_rank: Int,
    output_shape: DimList,
    input_rank: Int,
    input_shape: DimList,
    indices_rank: Int,
    indices_shape: DimList,
    type: DType,
    indices_type: DType,
    axis: Int,
    simd_width: Int,
](
    output: NDBuffer[output_rank, output_shape, type],
    input: NDBuffer[input_rank, input_shape, type],
    indices: NDBuffer[
        indices_rank,
        indices_shape,
        indices_type,
    ],
    out_chain: OutputChainPtr,
):
    """Computes output[d0, d1, ..., s0, s1, ..., sm, ..., dn-1, dn] =
    input[d0, d1, ..., indices[s0, s1, ..., sm], ..., dn-1, dn]"""
    assert_param[axis == input_rank - 1]()

    let outer_dynamic = prod_dims[0, axis](input)
    let indices_size = indices.size()
    let inner_dynamic = prod_dims[axis + 1, input_rank](input)

    @always_inline
    @parameter
    fn task_func(task_id: Int):
        for s in range(indices_size):
            let tuple_s = indices.get_nd_index(s)
            let idx: Int = indices[tuple_s].to_int()
            for do in range(outer_dynamic):
                for di in range(inner_dynamic):
                    output[
                        output.get_nd_index(
                            ((do * indices_size + s) * inner_dynamic) + di
                        )
                    ] = input[
                        input.get_nd_index(
                            ((do * input.dim[axis]() + idx) * inner_dynamic)
                            + di
                        )
                    ]

    async_parallelize[task_func](out_chain, 1)
