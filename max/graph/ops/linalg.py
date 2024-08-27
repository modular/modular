# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Linear algebra operations."""
from typing import Union

import numpy as np
from max.dtype import DType
from max.mlir.dialects import mo, rmo

from ..graph import Graph
from ..graph_value import GraphValue, ValueLike
from .constant import scalar


def matmul(lhs: ValueLike, rhs: ValueLike) -> GraphValue:
    """Computes the matrix multiplication of two tensor graph values.

    Performs general matrix multiplication with broadcasting.

    If the lhs is 1D, it will be reshaped to ``1xD``.
    If the rhs is 1D, it will be reshaped to ``Dx1``.
    In both cases, the additional ``1`` dimensions will be removed from the
    output shape.

    For the multiplication, the innermost (rightmost) 2 dimensions are treated
    as a matrix.
    The lhs matrix will have the shape ``MxK``.
    The rhs matrix will have the shape ``KxN``.
    The output will have the shape `MxN`
    The ``K`` dimensions must be equivalent in both matrices.

    The remaining outer dimensions will be broadcasted.

    Args:
        lhs: The left-hand-side of the matmul.
        rhs: The right-hand-side of the matmul.
        location: An optional location for a more specific error message.

    Returns:
        A tensor graph value representing he result of broadcasting the two
        matricies together and then performing a matrix multiply
        along the innermost two dimension of each tensor.
    """
    return Graph.current._add_op(rmo.matmul, GraphValue(lhs), GraphValue(rhs))[
        0
    ]


def band_part(
    x: ValueLike, num_lower: int, num_upper: int, exclude: bool = False
) -> GraphValue:
    """Masks out everything except a diagonal band of an input matrix.

    Copies a tensor setting everything outside the central diagonal band of the
    matricies to zero, where all but the last two axes are effectively batches,
    and the last two axes define sub matricies.

    Assumes the input has dimensions [I, J, ..., M, N], then the output tensor
    has the same shape as the input, and the values are given by

    .. code-block:: python

        out[i, j, ..., m, n] = in_band(m, n) * input[i, j,  ..., m, n].

    with the indicator function:

    .. code-block:: python

        in_band(m, n) = ((num_lower < 0 || (m - n) <= num_lower)) &&
                        (num_upper < 0 || (n - m) <= num_upper))

    Args:
        input: The input to mask out.
        num_lower: The number of diagonal bands to include below the central
            diagonal. If -1, include the entire lower triangle.
        num_upper: The number of diagonal bands to include above the central
            diagonal. If -1, include the entire upper triangle.
        exclude: If true, invert the selection of elements to mask. Elements
            in the band are set to zero.

    Returns:
        A symbolic tensor value with the configured selection masked out
        to 0 values, and the remaining values copied from the input tensor.
    """
    x = GraphValue(x)
    return Graph.current._add_op(
        rmo.mo_linalg_band_part,
        x.tensor_type.to_mlir(),
        x,
        scalar(num_lower, DType.int64),
        scalar(num_upper, DType.int64),
        scalar(exclude, DType.bool),
    )[0]


def layer_norm(
    input: GraphValue, gamma: ValueLike, beta: ValueLike, epsilon: float
) -> GraphValue:
    """Performs layer normalization.

    Args:
        input: The input tensor to normalize.
        gamma: The gamma parameter of the normalization.
        beta: The beta parameter of the normalization.
        epsilon: The epsilon parameter of the normalization.

    Returns:
        A graph tensor value with the normalization applied.
    """
    return Graph.current._add_op(
        mo.layer_norm,
        input._mlir_value.type,
        input,
        GraphValue(gamma),
        GraphValue(beta),
        GraphValue(np.array(epsilon)),
    )[0]
