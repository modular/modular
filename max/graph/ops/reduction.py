# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Reduction ops."""

import numpy as np
from max.mlir.dialects import rmo

from ..graph import Graph
from ..graph_value import GraphValue, ValueLike
from ..type import Dim, Shape, TensorType


def mean(x: ValueLike, axis=-1):
    gv = GraphValue(x)
    shape = Shape(gv.shape)
    shape[axis] = Dim(1)
    type = TensorType(gv.dtype, shape)
    return Graph.current._add_op(
        rmo.mo_mean, type.to_mlir(), gv, GraphValue(np.array(axis))
    )[0]
