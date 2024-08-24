# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Core graph primitives."""
from typing import Union

import numpy as np
from max import _graph
from max.dtype import DType
from max.mlir.dialects import mo

from ..graph import Graph
from ..value import TensorValue
from ..type import TensorType


def constant(value: np.ndarray) -> TensorValue:
    """Adds a node representing a constant operation.

    The value of this constant will have the type `TensorType` with the
    same shape and dtype as `value`.

    Parameters:
        dtype: The constant tensor's element type.

    Args:
        value: The constant's value.

    Returns:
        A graph value containing the constant data as an attribute.
    """
    tensor_type = TensorType(
        DType.from_numpy(value.dtype), value.shape
    ).to_mlir()
    array_attr = _graph.array_attr(
        "value", np.ascontiguousarray(value), tensor_type
    )
    return Graph.current._add_op(
        mo.constant, result=tensor_type, value=array_attr
    )[0].tensor


def scalar(
    value: Union[int, float], dtype: DType, rank: int = 0
) -> TensorValue:
    """Creates a scalar in the graph and returns the value.

    A scalar is a tensor with a single element. Generally scalars are of rank 0, but that is configurable.
    """
    tensor_type = TensorType(dtype, [1] * rank).to_mlir()
    array_attr = _graph.array_attr(
        "value", np.array([value]).astype(dtype.to_numpy()), tensor_type
    )
    return Graph.current._add_op(
        mo.constant, result=tensor_type, value=array_attr
    )[0].tensor
