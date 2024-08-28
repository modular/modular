# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for outer."""

from typing import Iterable

from max import mlir
from max.mlir.dialects import rmo

from ..graph import Graph
from ..value import TensorValue, ValueLike
from .reshape import reshape


def outer(lhs: ValueLike, rhs: ValueLike) -> TensorValue:
    """Computes the outer product of two symbolic vectors.

    Args:
        lhs: The left side of the product. Whatever its shape,
            it will be flattened to a rank-1 vector.
        rhs: The right side of the product. Whatever its shape,
            it will be flattened to a rank-1 vector. Must have the
            same number of elements as `lhs`.

    Returns:
        A symbolic tensor representing the
        [outer product](https://en.wikipedia.org/wiki/Outer_product)
        of the two input vectors. It will have rank 2, with the dimension
        sizes being the number of elements of `lhs` and `rhs` respectively.
    """
    return reshape(lhs, [-1, 1]) * reshape(rhs, [1, -1])
