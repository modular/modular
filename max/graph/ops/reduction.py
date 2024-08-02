# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Reduction ops."""

from max.mlir.dialects import rmo
import numpy as np

from ..graph import Graph
from ..graph_value import GraphValue, ValueLike
from ..type import dim


def mean(x: ValueLike, axis=-1):
    gv = GraphValue(x)
    type = gv.tensor_type
    type.shape[axis] = dim(1)
    return Graph.current._add_op(
        rmo.mo_mean, type.to_mlir(), gv, GraphValue(np.array(axis))
    )[0]
