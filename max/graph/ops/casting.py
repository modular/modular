# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Casting ops."""

import numpy as np
from max import mlir
from max.mlir.dialects import mo, rmo

from ..graph import Graph
from ..graph_value import GraphValue, TensorType, ValueLike, ops
from ..type import DType, Shape, ShapeLike, StaticDim


def rebind(x: ValueLike, shape: ShapeLike, message: str):
    # TODO(MSDK-662): Add checks to ensure that statically known dims are
    # rebound in a way to keep the size the same.
    v = GraphValue(x)
    message_attr = mlir.StringAttr.get(message)
    return Graph.current._add_op(
        rmo.rebind_tensor_shape,
        TensorType(v.dtype, shape).to_mlir(),
        v,
        message=message_attr,
    )[0]


def reshape(x: ValueLike, shape: ShapeLike):
    return Graph.current._add_op(
        rmo.reshape, GraphValue(x), new_shape=Shape(shape).to_mlir()
    )[0]


def cast(x: ValueLike, dtype: DType):
    gv = GraphValue(x)
    if gv.tensor_type.dtype == dtype:
        return gv
    return Graph.current._add_op(
        mo.cast, gv.tensor_type.cast(dtype).to_mlir(), gv
    )[0]


def unsqueeze(x: ValueLike, axis: int) -> GraphValue:
    x = GraphValue(x)
    rank = x.rank
    if axis < 0:
        axis += rank + 1
    if not 0 <= axis <= rank:
        raise ValueError(f"unsqueeze axis out of bounds: {axis=}, {rank=}")

    shape = x.shape
    new_shape = shape[:axis] + [1] + shape[axis:]
    return ops.reshape(x, new_shape)


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
            if isinstance(d, StaticDim) and d.dim != 1:
                new_shape.append(d)
    return ops.reshape(v, new_shape)


def transpose(x: ValueLike, dim_1: int, dim_2: int) -> GraphValue:
    v = GraphValue(x)

    rank = len(v.shape)
    if dim_1 < 0:
        dim_1 += rank
    if dim_2 < 0:
        dim_2 += rank

    new_shape = v.shape
    indices = np.array(range(len(new_shape)))

    new_shape[dim_1], new_shape[dim_2] = new_shape[dim_2], new_shape[dim_1]
    indices[dim_1], indices[dim_2] = dim_2, dim_1

    return Graph.current._add_op(
        rmo.mo_transpose,
        TensorType(dtype=v.tensor_type.dtype, shape=new_shape).to_mlir(),
        v,
        GraphValue(indices),
    )[0]


def broadcast_to(x: GraphValue, shape: ShapeLike):
    return Graph.current._add_op(
        rmo.broadcast_to, x, new_shape=Shape(shape).to_mlir()
    )[0]


def shape_to_tensor(shape: ShapeLike) -> GraphValue:
    return Graph.current._add_op(
        rmo.shape_to_tensor, shape=Shape(shape).to_mlir()
    )[0]
