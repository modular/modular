# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for reshape."""

import numpy as np
from max.mlir.dialects import rmo

from ..graph import Graph
from ..type import Shape, ShapeLike, StaticDim
from ..value import TensorValue, TensorValueLike


def reshape(x: TensorValueLike, shape: ShapeLike) -> TensorValue:
    """Reshapes a symbolic tensor.

    The number and order of the elements in the tensor is unchanged.
    In other words, if you were to iterate over elements in the tensor
    by major dimension to minor dimension, the iteration order would stay
    the same.

    If a value of -1 is present in the shape, that dimension becomes
    an automatically calculated dimension collecting all unspecified dimensions.
    Its length becomes the number of elements in the original tensor
    divided by the product of elements of the reshape.

    Args:
        x: The input symbolic tensor to reshape.
           This tensor may not contain any dynamic dimensions.
        shape: The new shape as a list of dimensions.
               Dynamic dimensions are not allowed.
               A single dimension may be `-1`.

    Returns:
        A symbolic tensor with the same elements as the original tensor, but
        in a new shape. Its symbolic shape is the same as :code:`shape`.

    Raises:
        ValueError: if input and target shapes' number of elements mismatch.
    """
    x = TensorValue(x)
    shape = Shape(shape)

    def is_static(dims):
        return all(isinstance(dim, StaticDim) and dim.dim >= 0 for dim in dims)

    # TODO(GRA-1015): Remove once static dims are int64 in the Graph compiler.
    if (
        # Leave checking shapes with -1 or symbolic dims to param expr folding.
        is_static(x.shape)
        and is_static(shape)
        and (np.prod(x.shape.static_dims) != np.prod(shape.static_dims))
    ):
        msg = (
            f"expected shapes {x} and {shape} to have the same number of "
            "elements"
        )
        raise ValueError(msg)

    return Graph.current._add_op(
        rmo.reshape, TensorValue(x), new_shape=Shape(shape).to_mlir()
    )[0].tensor
