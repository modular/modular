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


def mean(x: ValueLike, axis=-1):
    gv = GraphValue(x)
    type = gv.tensor_type
    type.shape[axis] = 1
    return Graph.current._add_op(
        rmo.mo_mean, type.to_mlir(), gv, GraphValue(np.array(axis))
    )[0]
