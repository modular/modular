# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param
from Buffer import NDBuffer
from DType import DType
from Functional import vectorize, vectorize_unroll, div_ceil, parallelForEachN
from Index import Index, StaticIntTuple
from Intrinsics import PrefetchOptions
from Int import Int
from List import create_kgen_list_unknown
from LLCL import Runtime
from Math import add
from Pointer import Pointer
from Range import range
from TargetInfo import dtype_sizeof
from TypeUtilities import rebind
from SIMD import SIMD

# This sets the min copy size per task because it's inefficient to copy
# small volume in parallel.
# TODO: find a heuristic to replace the magic number.
alias MIN_TASK_COPY_SIZE = 256 * 1024  # Bytes


# gather_reduce_2D_axis_1
@adaptive
fn gather_reduce[
    output_rank: __mlir_type.index,
    output_shape: __mlir_type[`!kgen.list<index[`, output_rank, `]>`],
    input_rank: __mlir_type.index,
    input_shape: __mlir_type[`!kgen.list<index[`, input_rank, `]>`],
    indices_rank: __mlir_type.index,
    indices_shape: __mlir_type[`!kgen.list<index[`, indices_rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
    gather_axis: __mlir_type.index,
    reduce_axis: __mlir_type.index,
    simd_width: __mlir_type.index,
    reduce_fn: __mlir_type[
        `!kgen.signature<<simd_width, type: dtype>(`,
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
        DType.si32.value,
    ],
    reduce_init: SIMD[1, type],
):
    """Computes output[i, j, k] = input[indices[i, j], k] and simultaneously
    reduces the output accross axis 1."""
    assert_param[output_rank == 2]()
    assert_param[input_rank == 2]()
    assert_param[indices_rank == 2]()
    assert_param[gather_axis == 0]()
    assert_param[reduce_axis == 1]()

    alias unroll_factor = 2
    alias usimd_width = simd_width * unroll_factor

    _ = output.fill(reduce_init)

    @always_inline
    fn _gather_contiguous[simd_width: __mlir_type.index](i: Int, j: Int):
        let idx = indices[i, j].value

        @always_inline
        fn _simd_gather[simd_width: __mlir_type.index](k: Int):
            let in_idx = StaticIntTuple[input_rank](idx, k)
            let out_idx = StaticIntTuple[output_rank](i, k)

            let gather_chunk = input.simd_load[simd_width](in_idx)
            let accum = output.simd_load[simd_width](out_idx)

            output.simd_store[simd_width](
                out_idx,
                reduce_fn[simd_width, type](accum, gather_chunk),
            )

        vectorize[simd_width, _simd_gather](output.dim[1]())

    for i in range(output.dim[0]()):
        for j in range(indices.dim[1]()):
            _gather_contiguous[usimd_width](i, j)


# Argument for gather task
struct gather_args[type: __mlir_type.`!kgen.dtype`]:
    var output: NDBuffer[2, create_kgen_list_unknown[2](), type]
    var input: NDBuffer[2, create_kgen_list_unknown[2](), type]
    var indices: NDBuffer[1, create_kgen_list_unknown[1](), DType.si32.value]
    var num_rows_per_task: Int

    fn __clone__(self&) -> Self:
        return Self {
            output: self.output,
            input: self.input,
            indices: self.indices,
            num_rows_per_task: self.num_rows_per_task,
        }

    fn __new__(
        output: NDBuffer[2, create_kgen_list_unknown[2](), type],
        input: NDBuffer[2, create_kgen_list_unknown[2](), type],
        indices: NDBuffer[1, create_kgen_list_unknown[1](), DType.si32.value],
        num_rows_per_task: Int,
    ) -> gather_args[type]:
        return gather_args[type] {
            output: output,
            input: input,
            indices: indices,
            num_rows_per_task: num_rows_per_task,
        }


# gather_2D_input_1D_indices_axis_0
@adaptive
fn gather[
    output_rank: __mlir_type.index,
    output_shape: __mlir_type[`!kgen.list<index[`, output_rank, `]>`],
    input_rank: __mlir_type.index,
    input_shape: __mlir_type[`!kgen.list<index[`, input_rank, `]>`],
    indices_rank: __mlir_type.index,
    indices_shape: __mlir_type[`!kgen.list<index[`, indices_rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
    axis: __mlir_type.index,
    simd_width: __mlir_type.index,
](
    output: NDBuffer[output_rank, output_shape, type],
    input: NDBuffer[input_rank, input_shape, type],
    indices: NDBuffer[
        indices_rank,
        indices_shape,
        DType.si32.value,
    ],
    runtime: Runtime.ptr_type,
):
    """Computes output[i, j] = input[indices[i], j]"""
    assert_param[output_rank == 2]()
    assert_param[input_rank == 2]()
    assert_param[indices_rank == 1]()
    assert_param[axis == 0]()

    let indices_len = indices.size()

    fn task_func(task_id: Int, ptr: Pointer[gather_args[type]]):
        let gather_args = ptr.load()
        let output = gather_args.output
        let input = gather_args.input
        let indices = gather_args.indices
        let num_rows_per_task = gather_args.num_rows_per_task

        let start_offset = task_id * num_rows_per_task
        let end_offset = Int.min(
            (task_id + 1) * num_rows_per_task, indices.size()
        )

        # TODO: Find a heuristic to remove magic number.
        let prefetch_offset = 6
        let row_size = input.dim[1]()

        for i in range(start_offset, end_offset):
            input.prefetch[
                PrefetchOptions().for_read().high_locality().to_data_cache()
            ]((indices[i + prefetch_offset]).value, 0)

            let output_row_ptr = output.data.offset(i * row_size)
            let input_row_ptr = input.data.offset((indices[i] * row_size).value)

            @always_inline
            fn func_wrapper[simd_width: __mlir_type.index](idx: Int):
                output_row_ptr.simd_store[simd_width](
                    idx, input_row_ptr.simd_load[simd_width](idx)
                )

            vectorize_unroll[simd_width, 2, func_wrapper](row_size)

    # Decide number of tasks.
    # Each task consists of several rows and we don't split rows. This will
    # need to be refactored if the input has only a few but very long rows.
    let min_task_num_rows = div_ceil(
        MIN_TASK_COPY_SIZE, input.dim[1]() * dtype_sizeof[type]()
    )
    let num_threads = Runtime(runtime).parallelism_level()
    let num_tasks = Int.min(
        div_ceil(indices_len, min_task_num_rows), num_threads
    )

    let num_rows_per_task = div_ceil(indices_len, num_tasks)
    let output_bind = rebind[NDBuffer[2, create_kgen_list_unknown[2](), type]](
        output
    )
    let input_bind = rebind[NDBuffer[2, create_kgen_list_unknown[2](), type]](
        input
    )
    let indices_bind = rebind[
        NDBuffer[1, create_kgen_list_unknown[1](), DType.si32.value]
    ](indices)
    var args = gather_args[type](
        output_bind, input_bind, indices_bind, num_rows_per_task
    )
    let args_address = Pointer[gather_args[type]].address_of(args)

    parallelForEachN[Pointer[gather_args[type]], task_func](
        runtime, num_tasks, args_address
    )


# gather_2D_input_1D_indices_axis_1
@adaptive
fn gather[
    output_rank: __mlir_type.index,
    output_shape: __mlir_type[`!kgen.list<index[`, output_rank, `]>`],
    input_rank: __mlir_type.index,
    input_shape: __mlir_type[`!kgen.list<index[`, input_rank, `]>`],
    indices_rank: __mlir_type.index,
    indices_shape: __mlir_type[`!kgen.list<index[`, indices_rank, `]>`],
    type: __mlir_type.`!kgen.dtype`,
    axis: __mlir_type.index,
    simd_width: __mlir_type.index,
](
    output: NDBuffer[output_rank, output_shape, type],
    input: NDBuffer[input_rank, input_shape, type],
    indices: NDBuffer[
        indices_rank,
        indices_shape,
        DType.si32.value,
    ],
    runtime: Runtime.ptr_type,
):
    """Computes output[i, j] = input[i, indices[j]]"""
    assert_param[output_rank == 2]()
    assert_param[input_rank == 2]()
    assert_param[indices_rank == 1]()
    assert_param[axis == 1]()

    for i in range(output.dim[0]()):
        for j in range(output.dim[1]()):
            let idx: Int = indices[j].value
            output.__setitem__(
                StaticIntTuple[output_rank](i, j),
                input[i, idx],
            )
