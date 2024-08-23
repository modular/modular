# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for layer_norm."""

import numpy as np
from max.mlir.dialects import mo

from ..graph import Graph
from ..graph_value import GraphValue, ValueLike


def layer_norm(
    input: GraphValue, gamma: ValueLike, beta: ValueLike, epsilon: float
) -> GraphValue:
    """Performs layer normalization.

    Args:
        input: The input tensor to normalize.
        gamma: The gamma parameter of the normalization.
        beta: The beta parameter of the normalization.
        epsilon: The epsilon parameter of the normalization.

    Returns:
        A graph tensor value with the normalization applied.
    """
    return Graph.current._add_op(
        mo.layer_norm,
        input._mlir_value.type,
        input,
        GraphValue(gamma),
        GraphValue(beta),
        GraphValue(np.array(epsilon)),
    )[0]
