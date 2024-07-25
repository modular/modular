# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Casting ops."""

from typing import Union

from max import mlir
from max.mlir.dialects import rmo

from .. import core as _c
from ..graph import Graph
from ..graph_value import GraphValue
from ..type import Dim, dim


def reshape(x: GraphValue, *dims: Union[int, str, Dim]):
    dims = [dim(d).to_mlir() for d in dims]
    return Graph.current._add_op(
        rmo.reshape, x, new_shape=_c.shape_attr(mlir.Context.current, dims)
    )[0]
