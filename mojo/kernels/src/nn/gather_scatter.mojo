# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param
from Buffer import NDBuffer
from DType import DType
from Functional import vectorize
from Index import Index, StaticIntTuple
from Intrinsics import _prefetch, PrefetchLocality, PrefetchRW, PrefetchCache
from Int import Int
from Range import range
from SIMD import SIMD
from Math import add
from Functional import vectorize


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


# gather_2D_axis_0
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
):
    """Computes output[i, j, k] = input[indices[i, j], k]"""
    assert_param[output_rank == 3]()
    assert_param[input_rank == 2]()
    assert_param[indices_rank == 2]()
    assert_param[axis == 0]()

    # TODO: Find a heuristic to remove magic number.
    let prefetch_offset = 6
    let row_size = input.dim[1]()

    # TODO: Clean up after issue #9080 fixed.
    # The function body can be moved inside the for loop.
    @always_inline
    fn gather_row(i: Int, j: Int):
        let input_row_idx = indices[
            StaticIntTuple[indices_rank](i, j),
        ].value
        let output_row_idx = i * indices.dim[1]() + j
        # Set the address to prefetch
        let ptr_next = input.data.offset(
            (
                indices[StaticIntTuple[indices_rank](i, j + prefetch_offset)]
                * row_size
            ).value
        )
        _prefetch[PrefetchRW.READ, PrefetchLocality.HIGH, PrefetchCache.DATA](
            ptr_next.bitcast[DType.invalid.value](),
        )

        let output_row_ptr = output.data.offset(output_row_idx * row_size)
        let input_row_ptr = input.data.offset(input_row_idx * row_size)

        @always_inline
        fn func_wrapper[simd_width: __mlir_type.index](idx: Int):
            output_row_ptr.simd_store[simd_width](
                idx, input_row_ptr.simd_load[simd_width](idx)
            )

        vectorize[simd_width, func_wrapper](row_size)

    for i in range(output.dim[0]()):
        for j in range(output.dim[1]()):
            gather_row(i, j)


# gather_2D_axis_1
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
):
    """Computes output[i, j, k] = input[i, indices[j, k]]"""
    assert_param[output_rank == 3]()
    assert_param[input_rank == 2]()
    assert_param[indices_rank == 2]()
    assert_param[axis == 1]()

    for i in range(output.dim[0]()):
        for j in range(output.dim[1]()):
            for k in range(output.dim[2]()):
                let idx: Int = indices[j, k].value
                output.__setitem__(
                    StaticIntTuple[output_rank](i, j, k),
                    input[i, idx],
                )


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
):
    """Computes output[i, j] = input[indices[i], j]"""
    assert_param[output_rank == 2]()
    assert_param[input_rank == 2]()
    assert_param[indices_rank == 1]()
    assert_param[axis == 0]()

    # TODO: Find a heuristic to remove magic number.
    let prefetch_offset = 6
    let row_size = input.dim[1]()

    # TODO: Clean up after issue #9080 fixed.
    # The function body can be moved inside the for loop.
    @always_inline
    fn gather_row(i: Int):
        let ptr_next = input.data.offset(
            (indices[i + prefetch_offset] * row_size).value
        )
        _prefetch[PrefetchRW.READ, PrefetchLocality.HIGH, PrefetchCache.DATA](
            ptr_next.bitcast[DType.invalid.value](),
        )

        let output_row_ptr = output.data.offset(i * row_size)
        let input_row_ptr = input.data.offset((indices[i] * row_size).value)

        @always_inline
        fn func_wrapper[simd_width: __mlir_type.index](idx: Int):
            output_row_ptr.simd_store[simd_width](
                idx, input_row_ptr.simd_load[simd_width](idx)
            )

        vectorize[simd_width, func_wrapper](row_size)

    for i in range(output.dim[0]()):
        gather_row(i)


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
