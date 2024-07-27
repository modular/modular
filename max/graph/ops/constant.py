# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Core graph primitives."""
import numpy as np
from max import _graph
from max.mlir.dialects import mo

from ..dtype import DType
from ..graph import Graph
from ..graph_value import GraphValue
from ..type import TensorType


def constant(value: np.ndarray) -> GraphValue:
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
    array_attr = _graph.array_attr("value", value, tensor_type)
    return Graph.current._add_op(
        mo.constant, result=tensor_type, value=array_attr
    )[0]
