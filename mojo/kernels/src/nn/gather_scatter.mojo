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


@interface
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
    ...


@implements(gather)
fn gather_2D_axis_0[
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
                        StaticIntTuple[3],
                        StaticIntTuple[output_rank],
                    ](Index(i, j, k)),
                    input[idx, k],
                )


@implements(gather)
fn gather_2D_axis_1[
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
                        StaticIntTuple[3],
                        StaticIntTuple[output_rank],
                    ](Index(i, j, k)),
                    input[i, idx],
                )


@implements(gather)
fn gather_2D_input_1D_indices_axis_0[
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
                    StaticIntTuple[2],
                    StaticIntTuple[output_rank],
                ](Index(i, j)),
                input[idx, j],
            )


@implements(gather)
fn gather_2D_input_1D_indices_axis_1[
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
                    StaticIntTuple[2],
                    StaticIntTuple[output_rank],
                ](Index(i, j)),
                input[i, idx],
            )
