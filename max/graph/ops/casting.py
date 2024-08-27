# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Casting ops."""

import numpy as np
from max import mlir
from max.dtype import DType
from max.mlir.dialects import mo, rmo

from ..graph import Graph
from ..graph_value import GraphValue, TensorType, ValueLike, ops
from ..type import Shape, ShapeLike, StaticDim


def rebind(x: ValueLike, shape: ShapeLike, message: str):
    """Rebinds a symbolic tensor to a specified set of dimensions.

    This does not mutate the symbolic tensor passed in, but instead adds a
    runtime assert that the input symbolic shape is equivalent to `out_dims`
    shape. For example, if the input tensor shape has dynamic/unknown sizes,
    this will assert a fixed sizes that may be required for a subsequent
    operation.

    Args:
        x: The input symbolic tensor to rebind.
        shape: The symbolic shape to assert for `x`, as a list of
                  [`Dim`](/max/api/mojo/graph/type/Dim) values.
        message: The message printed if the rebind fails at runtime.

    Returns:
        A symbolic tensor with the same elements and shape as the given
        tensor, but with the symbolic shape asserted to `shape`.

    """
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


def squeeze(x: ValueLike, axis: int) -> GraphValue:
    v = GraphValue(x)
    # TODO (MSDK-655): Probably want to add rmo.mo_squeeze_shape here
    shape = Shape(v.shape)
    if shape[axis] != 1:
        raise ValueError(f"Squeeze dim must be 1, got {axis=}, {shape=}")
    shape.pop(axis)
    return ops.reshape(v, shape)


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
