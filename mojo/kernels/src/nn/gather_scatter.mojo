# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param, assert_param_bool, debug_assert
from Buffer import NDBuffer
from DType import DType
from Functional import vectorize, vectorize_unroll, parallelize
from Math import div_ceil
from Index import Index, StaticIntTuple
from Intrinsics import PrefetchOptions
from Int import Int
from List import create_kgen_list_unknown
from LLCL import Runtime
from Math import add, min
from Pointer import Pointer
from Range import range
from TargetInfo import dtype_sizeof
from TypeUtilities import rebind
from SIMD import SIMD


## gather_reduce_2D_axis_1
@adaptive
fn gather_reduce[
    output_rank: __mlir_type.index,
    output_shape: __mlir_type[`!kgen.list<index[`, output_rank, `]>`],
    input_rank: __mlir_type.index,
    input_shape: __mlir_type[`!kgen.list<index[`, input_rank, `]>`],
    indices_rank: __mlir_type.index,
    indices_shape: __mlir_type[`!kgen.list<index[`, indices_rank, `]>`],
    type: DType,
    gather_axis: __mlir_type.index,
    reduce_axis: __mlir_type.index,
    simd_width: Int,
    reduce_fn: __mlir_type[
        `!kgen.signature<<simd_width:`,
        Int,
        `, type: `,
        DType,
        `>(`,
        SIMD[simd_width, `type`],
        `,`,
        SIMD[simd_width, `type`],
        `) ->`,
        SIMD[simd_width, `type`],
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
    runtime: Runtime.ptr_type,
):
    """Computes output[i, j, k] = input[indices[i, j], k] and simultaneously
    reduces the output accross axis 1 to produce output[i, k].

    The motivating use-case for this is multi-hot embeddings in recommender models.
    This provides similar functionality to Torch's EmbeddingBag layer. In that
    context, i is the batch dimension, j is the multi-hot dimension, and k is
    the embedding dimension.
    """
    assert_param[output_rank == 2]()
    assert_param[input_rank == 2]()
    assert_param[indices_rank == 2]()
    assert_param[gather_axis == 0]()
    assert_param[reduce_axis == 1]()

    # TODO: find a heuristic to replace the magic number.
    # This is about 4x larger than the default in gather, which makes sense
    # since this kernel performs far fewer writes
    alias MIN_TASK_COPY_SIZE = 64 * 100 * 32 * 4  # bytes
    let num_threads = Runtime(runtime).parallelism_level()
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
    let output_bind = rebind[NDBuffer[2, create_kgen_list_unknown[2](), type]](
        output
    )
    let input_bind = rebind[NDBuffer[2, create_kgen_list_unknown[2](), type]](
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

    parallelize[task_func](runtime, num_tasks)


# gather_2D_input_1D_indices_axis_0
@adaptive
fn gather[
    output_rank: __mlir_type.index,
    output_shape: __mlir_type[`!kgen.list<index[`, output_rank, `]>`],
    input_rank: __mlir_type.index,
    input_shape: __mlir_type[`!kgen.list<index[`, input_rank, `]>`],
    indices_rank: __mlir_type.index,
    indices_shape: __mlir_type[`!kgen.list<index[`, indices_rank, `]>`],
    type: DType,
    indices_type: DType,
    axis: Int,
    simd_width: Int,
](
    output: NDBuffer[output_rank, output_shape, type],
    input: NDBuffer[input_rank, input_shape, type],
    indices: NDBuffer[indices_rank, indices_shape, indices_type],
    runtime: Runtime.ptr_type,
):
    """Computes output[i, j] = input[indices[i], j]"""
    assert_param[output_rank == 2]()
    assert_param[input_rank == 2]()
    assert_param[indices_rank == 1]()
    assert_param_bool[axis == 0]()

    let indices_len = indices.size()

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
    let num_threads = Runtime(runtime).parallelism_level()
    let num_tasks = min(div_ceil(indices_len, min_task_num_rows), num_threads)

    let num_chunks_per_task = div_ceil(indices_len, num_tasks)
    let output_bind = rebind[NDBuffer[2, create_kgen_list_unknown[2](), type]](
        output
    )
    let input_bind = rebind[NDBuffer[2, create_kgen_list_unknown[2](), type]](
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
            ](
                (
                    Int.from_integral[indices_type](
                        indices[i + prefetch_offset].value
                    )
                ).value,
                0,
            )

            let output_row_ptr = output.data.offset(i * row_size)
            let input_row_ptr = input.data.offset(
                (
                    Int.from_integral[indices_type](indices[i].value) * row_size
                ).value
            )

            @always_inline
            fn func_wrapper[simd_width: Int](idx: Int):
                output_row_ptr.simd_store[simd_width](
                    idx, input_row_ptr.simd_load[simd_width](idx)
                )

            vectorize_unroll[simd_width, 2, func_wrapper](row_size)

    parallelize[task_func](runtime, num_tasks)


# gather_2D_input_1D_indices_axis_1
@adaptive
fn gather[
    output_rank: __mlir_type.index,
    output_shape: __mlir_type[`!kgen.list<index[`, output_rank, `]>`],
    input_rank: __mlir_type.index,
    input_shape: __mlir_type[`!kgen.list<index[`, input_rank, `]>`],
    indices_rank: __mlir_type.index,
    indices_shape: __mlir_type[`!kgen.list<index[`, indices_rank, `]>`],
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
    runtime: Runtime.ptr_type,
):
    """Computes output[i, j] = input[i, indices[j]]"""
    assert_param[output_rank == 2]()
    assert_param[input_rank == 2]()
    assert_param[indices_rank == 1]()
    assert_param_bool[axis == 1]()

    for i in range(output.dim[0]()):
        for j in range(output.dim[1]()):
            let idx: Int = Int.from_integral[indices_type](indices[j].value)
            output[StaticIntTuple[output_rank](i, j)] = input[i, idx]
