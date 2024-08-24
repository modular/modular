# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Reduction ops."""

import numpy as np
from max.mlir.dialects import rmo

from ..graph import Graph
from ..value import TensorValue, ValueLike
from ..type import Dim, Shape, TensorType


def mean(x: ValueLike, axis=-1) -> TensorValue:
    gv = TensorValue(x)
    shape = Shape(gv.shape)
    shape[axis] = Dim(1)
    type = TensorType(gv.dtype, shape)
    return Graph.current._add_op(
        rmo.mo_mean, type.to_mlir(), gv, TensorValue(np.array(axis))
    )[0].tensor
