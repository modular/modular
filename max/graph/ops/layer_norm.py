# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for layer_norm."""

import numpy as np
from max.mlir.dialects import mo

from ..graph import Graph
from ..value import TensorValue, ValueLike


def layer_norm(
    input: TensorValue, gamma: ValueLike, beta: ValueLike, epsilon: float
) -> TensorValue:
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
        TensorValue(gamma),
        TensorValue(beta),
        TensorValue(np.array(epsilon)),
    )[0].tensor
