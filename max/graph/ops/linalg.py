# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Linear algebra operations."""
import numpy as np
from typing import Union
from max.mlir.dialects import mo, rmo

from ..graph import Graph
from ..graph_value import GraphValue, ValueLike


def matmul(lhs: ValueLike, rhs: ValueLike) -> GraphValue:
    """Computes the matrix multiplication of two tensor graph values.

    Performs general matrix multiplication with broadcasting.

    If the lhs is 1D, it will be reshaped to `1xD`.
    If the rhs is 1D, it will be reshaped to `Dx1`.
    In both cases, the additional `1` dimensions will be removed from the
    output shape.

    For the multiplication, the innermost (rightmost) 2 dimensions are treated
    as a matrix.
    The lhs matrix will have the shape `MxK`.
    The rhs matrix will have the shape `KxN`.
    The output will have the shape `MxN`
    The `K` dimensions must be equivalent in both matrices.

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
