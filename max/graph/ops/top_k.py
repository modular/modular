# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for top_k."""

from max.mlir.dialects import rmo

from ..graph import Graph
from ..value import TensorValue, TensorValueLike


def top_k(
    input: TensorValueLike, k: int, axis: int = -1
) -> tuple[TensorValue, TensorValue]:
    """Returns tensor with only top K values along given axis.

    Args:
        input: The input tensor from which to select top k.
        k: The number of values to select from input.
        axis: The axis from which to select top k.

    Returns:
        Top K values, Top K indices
    """
    topk_weight, topk_idx = Graph.current._add_op(
        rmo.top_k, TensorValue(input), k, axis
    )

    return topk_weight.tensor, topk_idx.tensor
