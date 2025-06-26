# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for matmul."""

from max.mlir.dialects import rmo

from ..graph import Graph
from ..value import TensorValue, TensorValueLike


def matmul(lhs: TensorValueLike, rhs: TensorValueLike) -> TensorValue:
    """Computes the matrix multiplication of two tensor graph values.

    Performs general matrix multiplication with broadcasting.

    If the lhs is 1D, it will be reshaped to ``1xD``.
    If the rhs is 1D, it will be reshaped to ``Dx1``.
    In both cases, the additional `1` dimensions will be removed from the
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
        matrices together and then performing a matrix multiply
        along the innermost two dimension of each tensor.
    """
    return Graph.current._add_op(
        rmo.matmul, TensorValue(lhs), TensorValue(rhs)
    )[0].tensor
