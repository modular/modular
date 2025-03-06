# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for layer_norm."""

from max.dtype import DType
from max.mlir.dialects import mo

from .. import dtype_promotion
from ..graph import Graph
from ..value import TensorValue, TensorValueLike
from .constant import constant


def layer_norm(
    input: TensorValue,
    gamma: TensorValueLike,
    beta: TensorValueLike,
    epsilon: float,
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
    input, gamma = dtype_promotion._promote_weak_dtypes(input, gamma)
    input, beta = dtype_promotion._promote_weak_dtypes(input, beta)
    return Graph.current._add_op(
        mo.layer_norm,
        input._mlir_value.type,
        input,
        gamma,
        beta,
        constant(epsilon, DType.float32),
    )[0].tensor
