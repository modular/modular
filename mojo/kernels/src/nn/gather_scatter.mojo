# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param
from Buffer import NDBuffer
from DType import DType
from Index import Index
from Int import Int
from Tuple import StaticTuple
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

    let i = output.dim[0]()
    let j = output.dim[1]()
    let k = output.dim[2]()

    var iter0: Int = 0
    while iter0 < i:
        var iter1: Int = 0
        while iter1 < j:
            var iter2: Int = 0
            let idx: Int = indices.__getitem__(
                rebind[
                    StaticTuple[
                        2,
                        __mlir_type.index,
                    ],
                    StaticTuple[
                        indices_rank,
                        __mlir_type.index,
                    ],
                ](Index(iter0, iter1).as_tuple())
            ).value
            while iter2 < k:
                output.__setitem__(
                    rebind[
                        StaticTuple[
                            3,
                            __mlir_type.index,
                        ],
                        StaticTuple[
                            output_rank,
                            __mlir_type.index,
                        ],
                    ](Index(iter0, iter1, iter2).as_tuple()),
                    input.__getitem__(
                        rebind[
                            StaticTuple[
                                2,
                                __mlir_type.index,
                            ],
                            StaticTuple[
                                input_rank,
                                __mlir_type.index,
                            ],
                        ](Index(idx, iter2).as_tuple())
                    ),
                )
                iter2 += 1
            iter1 += 1
        iter0 += 1


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

    let i = output.dim[0]()
    let j = output.dim[1]()
    let k = output.dim[2]()

    var iter0: Int = 0
    while iter0 < i:
        var iter1: Int = 0
        while iter1 < j:
            var iter2: Int = 0
            while iter2 < k:
                let idx: Int = indices.__getitem__(
                    rebind[
                        StaticTuple[
                            2,
                            __mlir_type.index,
                        ],
                        StaticTuple[
                            indices_rank,
                            __mlir_type.index,
                        ],
                    ](Index(iter1, iter2).as_tuple())
                ).value
                output.__setitem__(
                    rebind[
                        StaticTuple[
                            3,
                            __mlir_type.index,
                        ],
                        StaticTuple[
                            output_rank,
                            __mlir_type.index,
                        ],
                    ](Index(iter0, iter1, iter2).as_tuple()),
                    input.__getitem__(
                        rebind[
                            StaticTuple[
                                2,
                                __mlir_type.index,
                            ],
                            StaticTuple[
                                input_rank,
                                __mlir_type.index,
                            ],
                        ](Index(iter0, idx).as_tuple())
                    ),
                )
                iter2 += 1
            iter1 += 1
        iter0 += 1
