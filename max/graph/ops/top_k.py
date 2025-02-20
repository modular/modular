# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for top_k."""

from max.dtype import DType
from max.mlir.dialects import mo

from ..graph import Graph
from ..value import TensorType, TensorValue, TensorValueLike
from .constant import constant


def top_k(
    input: TensorValueLike,
    k: int,
    axis: int = -1,
    sorted: bool = False,
) -> tuple[TensorValue, TensorValue]:
    """Returns tensor with only top K values along given axis.

    Args:
        input: The input tensor from which to select top k.
        k: The number of values to select from input.
        axis: The axis from which to select top k.
        sorted: returns tensor sort by k if True.

    Returns:
        Top K values, Top K indices
    """

    input = TensorValue(input)
    input_shape = input.shape

    if axis < 0:
        axis += len(input_shape)  # Handle negative axis

    output_shape = input_shape.static_dims
    output_shape[axis] = k  # Replace the specified dim with k

    topk_weight, topk_idx = Graph.current._add_op(
        mo.top_k,
        TensorType(input.dtype, output_shape, input.device).to_mlir(),
        TensorType(DType.int64, output_shape, input.device).to_mlir(),
        TensorValue(input),
        constant(k, DType.int64),
        constant(axis, DType.int64),
        constant(sorted, DType.bool),
    )

    topk_weight_tensor, topk_idx_tensor = topk_weight.tensor, topk_idx.tensor
    assert isinstance(topk_weight_tensor, TensorValue)
    assert isinstance(topk_idx_tensor, TensorValue)

    return topk_weight_tensor, topk_idx_tensor
