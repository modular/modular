# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Reduction ops."""

import numpy as np
from max.mlir.dialects import rmo
from max.dtype import DType

from .constant import scalar
from ..graph import Graph
from ..value import TensorValue, ValueLike
from ..type import Dim, Shape, TensorType


def mean(x: ValueLike, axis=-1) -> TensorValue:
    """
    Reduces a symbolic tensor using a mean operation.

    Args:
        v: The input tensor for the operation.
        axis: The axis along which to compute the reduction. If negative,
            indexes from the last dimension. For example, a value of -1 will
            compute the reduction along the last dimension.

    Returns:
        A symbolic tensor representing the result of the mean operation.
        The tensor will have the same rank as the input tensor, and the same
        shape except along the ``axis`` dimension which will have size 1.
    """
    gv = TensorValue(x)
    shape = Shape(gv.shape)
    shape[axis] = Dim(1)
    type = TensorType(gv.dtype, shape)
    return Graph.current._add_op(
        rmo.mo_mean, type.to_mlir(), gv, scalar(axis, DType.int64)
    )[0].tensor
