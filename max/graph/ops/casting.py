# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Casting ops."""

from max import _graph, mlir
from max.mlir.dialects import rmo, mo
import numpy as np

from ..graph import Graph
from ..graph_value import GraphValue, ValueLike, TensorType, ops
from ..type import ShapeLike, dim, DType


def reshape(x: GraphValue, shape: ShapeLike):
    dims = [dim(d).to_mlir() for d in shape]
    return Graph.current._add_op(
        rmo.reshape, x, new_shape=_graph.shape_attr(mlir.Context.current, dims)
    )[0]


def cast(x: ValueLike, dtype: DType):
    gv = GraphValue(x)
    if gv.tensor_type.dtype == dtype:
        return gv
    return Graph.current._add_op(
        mo.cast, gv.tensor_type.cast(dtype).to_mlir(), gv
    )[0]


def unsqueeze(x: ValueLike, axis: int) -> GraphValue:
    v = GraphValue(x)
    rank = v.tensor_type.rank()
    if axis < 0:
        axis += rank + 1
    if axis < 0 or axis > rank:
        raise ValueError(
            "unsqueeze axis out of bounds: axis="
            + str(axis)
            + ", rank="
            + str(rank),
        )

    shape = v.tensor_type.shape
    new_shape = shape[:axis] + [1] + shape[axis:]
    return ops.reshape(v, new_shape)


def squeeze(x: ValueLike, axis=None) -> GraphValue:
    v = GraphValue(x)
    # TODO (MSDK-655): Probably want to add rmo.mo_squeeze_shape here
    if axis is not None:
        shape = v.tensor_type.shape
        if not shape[axis].is_static() or shape[axis].dim != 1:
            raise ValueError(
                f"Cannot squeeze axis {axis} with size"
                f" {v.tensor_type.shape[axis]}"
            )
        new_shape = v.tensor_type.shape[:axis] + v.tensor_type.shape[axis + 1 :]
    else:
        new_shape = []
        for d in v.tensor_type.shape:
            if d.is_static() and d.dim != 1:
                new_shape.append(d)
    return ops.reshape(v, new_shape)


# TODO (MSDK-655): This is a goofy, temporary implementation of transpose() to unblock llama development. Please replace with something smart.
def transpose(x: ValueLike, dim_1: int, dim_2: int) -> GraphValue:
    v = GraphValue(x)

    # TODO: needs negative index handling

    new_shape = v.tensor_type.shape
    new_shape[dim_1], new_shape[dim_2] = new_shape[dim_2], new_shape[dim_1]

    return Graph.current._add_op(
        rmo.mo_transpose,
        TensorType(dtype=v.tensor_type.dtype, shape=new_shape).to_mlir(),
        v,
        GraphValue(np.array([dim_1, dim_2])),
    )[0]


def broadcast_to(x: GraphValue, shape: ShapeLike):
    dims = [dim(d).to_mlir() for d in shape]
    return Graph.current._add_op(
        rmo.broadcast_to,
        x,
        new_shape=_graph.shape_attr(mlir.Context.current, dims),
    )[0]
