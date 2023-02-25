# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param
from Buffer import NDBuffer
from DType import DType
from Index import Index, StaticIntTuple
from Int import Int
from Range import range
from TypeUtilities import rebind
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
            let in_idx = rebind[StaticIntTuple[input_rank]](Index(idx, k))
            let gather_chunk = input.simd_load[simd_width](in_idx)
            let out_idx = rebind[StaticIntTuple[output_rank]](Index(i, k))
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

    for i in range(output.dim[0]()):
        for j in range(output.dim[1]()):
            let idx: Int = indices[i, j].value
            for k in range(output.dim[2]()):
                output.__setitem__(
                    rebind[
                        StaticIntTuple[output_rank],
                    ](Index(i, j, k)),
                    input[idx, k],
                )


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
                    rebind[
                        StaticIntTuple[output_rank],
                    ](Index(i, j, k)),
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

    for i in range(output.dim[0]()):
        for j in range(output.dim[1]()):
            let idx: Int = indices[i].value
            output.__setitem__(
                rebind[
                    StaticIntTuple[output_rank],
                ](Index(i, j)),
                input[idx, j],
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
                rebind[
                    StaticIntTuple[output_rank],
                ](Index(i, j)),
                input[i, idx],
            )
