# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param, assert_param_bool, debug_assert
from Buffer import NDBuffer
from DType import DType
from Functional import vectorize, vectorize_unroll, async_parallelize
from Math import div_ceil
from Index import Index, StaticIntTuple
from Intrinsics import PrefetchOptions
from List import Dim, DimList
from LLCL import Runtime, OutputChainPtr
from Math import add, min
from Pointer import Pointer
from Range import range
from TargetInfo import dtype_sizeof
from TypeUtilities import rebind
from SIMD import SIMD


## gather_reduce_2D_axis_1
@adaptive
fn gather_reduce[
    output_rank: Int,
    output_shape: DimList[output_rank],
    input_rank: Int,
    input_shape: DimList[input_rank],
    indices_rank: Int,
    indices_shape: DimList[indices_rank],
    type: DType,
    gather_axis: Int,
    reduce_axis: Int,
    simd_width: Int,
    reduce_fn: __mlir_type[
        `!kgen.signature<<`,
        Int,
        `,`,
        DType,
        `>(`,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType],
        ],
        ` borrow,`,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType],
        ],
        ` borrow) ->`,
        SIMD[
            __mlir_attr[`#kgen.param.index.ref<0, false, 0> : `, Int],
            __mlir_attr[`#kgen.param.index.ref<0, false, 1> : `, DType],
        ],
        `>`,
    ],
](
    output: NDBuffer[output_rank, output_shape, type],
    input: NDBuffer[input_rank, input_shape, type],
    indices: NDBuffer[
        indices_rank,
        indices_shape,
        DType.si32,
    ],
    reduce_init: SIMD[1, type],
    out_chain: OutputChainPtr,
):
    """Computes output[i, j, k] = input[indices[i, j], k] and simultaneously
    reduces the output accross axis 1 to produce output[i, k].

    The motivating use-case for this is multi-hot embeddings in recommender models.
    This provides similar functionality to Torch's EmbeddingBag layer. In that
    context, i is the batch dimension, j is the multi-hot dimension, and k is
    the embedding dimension.
    """
    assert_param_bool[input_rank == 2]()
    assert_param_bool[indices_rank == 2]()
    assert_param_bool[gather_axis == 0]()
    assert_param_bool[reduce_axis == 1]()

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

    let output_bind = NDBuffer[2, DimList[2].create_unknown(), type](
        output.data, output_2d_dims, type
    )
    let input_bind = rebind[NDBuffer[2, DimList[2].create_unknown(), type]](
        input
    )
    let indices_bind = rebind[
        NDBuffer[indices_rank, indices_shape, DType.si32]
    ](indices)

    @always_inline
    fn task_func(task_id: Int):
        alias unroll_factor = 2
        alias prefetch_offset = 6
        alias usimd_width = unroll_factor * simd_width

        let output = output_bind
        let input = input_bind
        let indices = indices_bind
        let row_size = output.dim[1]()

        # need to reduce on an entire 2D slice at a time, otherwise multiple
        # threads will try to accumulate in the same buffer simaltaneously
        let start_slice = task_id * num_chunks_per_task
        let end_slice = min(
            (task_id + 1) * num_chunks_per_task, indices.dim[0]()
        )

        for i in range(start_slice, end_slice):

            @always_inline
            fn _accum_in_place[simd_width: Int](k: Int):
                var accum = SIMD[simd_width, type](reduce_init)
                for j in range(indices.dim[1]()):
                    """Computes output[i,k] = reduction over j of (input[indices[i,j],k])
                    for j in range [0,indices.dim[1])"""
                    let idx = indices[i, j].value

                    # prefetch next k
                    let next_idx = indices._offset(
                        StaticIntTuple[indices_rank](i, j)
                    ).load()
                    input.prefetch[
                        PrefetchOptions()
                        .for_read()
                        .high_locality()
                        .to_data_cache()
                    ]((next_idx + prefetch_offset).value, 0)

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
    output_shape: DimList[output_rank],
    input_rank: Int,
    input_shape: DimList[input_rank],
    indices_rank: Int,
    indices_shape: DimList[indices_rank],
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
    assert_param_bool[output_rank == 2]()
    assert_param_bool[input_rank == 2]()
    assert_param_bool[indices_rank == 1]()
    assert_param_bool[axis == 0]()

    let indices_len = indices.size()
    # Short-circuit for trivial cases, and to avoid divide-by-zero
    if input.size() == 0 or indices_len == 0:
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
    let output_bind = rebind[NDBuffer[2, DimList[2].create_unknown(), type]](
        output
    )
    let input_bind = rebind[NDBuffer[2, DimList[2].create_unknown(), type]](
        input
    )

    @always_inline
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
    output_shape: DimList[output_rank],
    input_rank: Int,
    input_shape: DimList[input_rank],
    indices_rank: Int,
    indices_shape: DimList[indices_rank],
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
    """Computes output[i, j] = input[i, indices[j]]"""
    assert_param_bool[output_rank == 2]()
    assert_param_bool[input_rank == 2]()
    assert_param_bool[indices_rank == 1]()
    assert_param_bool[axis == 1]()

    for i in range(output.dim[0]()):
        for j in range(output.dim[1]()):
            let idx: Int = indices[j].to_int()
            output[StaticIntTuple[output_rank](i, j)] = input[i, idx]

    out_chain.mark_ready()
