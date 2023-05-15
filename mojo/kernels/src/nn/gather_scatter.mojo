# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param
from Buffer import NDBuffer, prod_dims
from DType import DType
from Functional import vectorize, vectorize_unroll, async_parallelize
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
    reduce_fn: fn[width: Int, type: DType] (
        SIMD[type, width], SIMD[type, width]
    ) -> SIMD[type, width],
](
    output: NDBuffer[output_rank, output_shape, type],
    input: NDBuffer[input_rank, input_shape, type],
    indices: NDBuffer[
        indices_rank,
        indices_shape,
        DType.si32,
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
        NDBuffer[indices_rank, indices_shape, DType.si32]
    ](indices)

    @always_inline
    @parameter
    fn task_func(task_id: Int):
        alias unroll_factor = 2
        alias prefetch_offset = 6
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

                    # prefetch next k
                    let next_idx_ptr = indices._offset(
                        StaticIntTuple[indices_rank](i, j)
                    ) + prefetch_offset
                    input.prefetch[
                        PrefetchOptions()
                        .for_read()
                        .high_locality()
                        .to_data_cache()
                    ](next_idx_ptr.load().value, 0)

                    let in_idx = StaticIntTuple[2](idx, k)

                    let gather_chunk = input.simd_load[simd_width](in_idx)
                    accum = reduce_fn[simd_width, type](accum, gather_chunk)

                let out_idx = StaticIntTuple[2](i, k)
                output.simd_store[simd_width](out_idx, accum)

            vectorize[usimd_width, _accum_in_place](row_size)

    async_parallelize[task_func](out_chain, num_tasks)


# gather_2D_input_1D_indices_axis_0
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
    """Computes output[i, j] = input[indices[i], j]"""
    assert_param[output_rank == 2]()
    assert_param[input_rank == 2]()
    assert_param[indices_rank == 1]()
    assert_param[axis == 0]()

    let indices_len = indices.size()
    # Short-circuit for trivial cases, and to avoid divide-by-zero
    if input.size() == 0 or indices_len == 0:
        # No-op
        out_chain.mark_ready()
        return

    # This sets the min copy size per task because it's inefficient to copy
    # small volume in parallel.
    # TODO: find a heuristic to replace the magic number.
    alias MIN_TASK_COPY_SIZE = 256 * 1024  # Bytes

    # Decide number of tasks.
    # Each task consists of several rows and we don't split rows. This will
    # need to be refactored if the input has only a few but very long rows.
    let min_task_num_rows = div_ceil(
        MIN_TASK_COPY_SIZE, input.dim[1]() * dtype_sizeof[type]()
    )
    let num_threads = out_chain.get_runtime().parallelism_level()
    let num_tasks = min(div_ceil(indices_len, min_task_num_rows), num_threads)

    let num_chunks_per_task = div_ceil(indices_len, num_tasks)
    let output_bind = rebind[NDBuffer[2, DimList.create_unknown[2](), type]](
        output
    )
    let input_bind = rebind[NDBuffer[2, DimList.create_unknown[2](), type]](
        input
    )

    @always_inline
    @parameter
    fn task_func(task_id: Int):
        let output = output_bind
        let input = input_bind

        let start_offset = task_id * num_chunks_per_task
        let end_offset = min(
            (task_id + 1) * num_chunks_per_task, indices.size()
        )

        # TODO: Find a heuristic to remove magic number.
        let prefetch_offset = 6
        let row_size = input.dim[1]()

        for i in range(start_offset, end_offset):
            input.prefetch[
                PrefetchOptions().for_read().high_locality().to_data_cache()
            ](indices[i + prefetch_offset].to_int(), 0)

            let output_row_ptr = output.data.offset(i * row_size)
            let input_row_ptr = input.data.offset(
                indices[i].to_int() * row_size
            )

            @always_inline
            @parameter
            fn func_wrapper[simd_width: Int](idx: Int):
                output_row_ptr.simd_store[simd_width](
                    idx, input_row_ptr.simd_load[simd_width](idx)
                )

            vectorize_unroll[simd_width, 2, func_wrapper](row_size)

    async_parallelize[task_func](out_chain, num_tasks)


# gather_2D_input_1D_indices_axis_1
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
    """Computes output[i, j] = input[i, indices[j]]"""
    assert_param[output_rank == 2]()
    assert_param[input_rank == 2]()
    assert_param[indices_rank == 1]()
    assert_param[axis == 1]()

    @always_inline
    @parameter
    fn task_func(task_id: Int):
        for i in range(output.dim[0]()):
            for j in range(output.dim[1]()):
                let idx: Int = indices[j].to_int()
                output[StaticIntTuple[output_rank](i, j)] = input[i, idx]

    async_parallelize[task_func](out_chain, 1)


# gather_ND_input_MD_indices
@adaptive
fn gather_nd[
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
