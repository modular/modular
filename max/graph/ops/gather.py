# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for gather."""


from max.dtype import DType
from max.mlir.dialects import rmo

from ..graph import Graph
from ..type import TensorType
from ..value import TensorValue, TensorValueLike
from .constant import constant


def gather(
    input: TensorValueLike, indices: TensorValueLike, axis: int = -1
) -> TensorValue:
    """
    Selects elements out of an input tensor by index.

    Args:
        input: The input symbolic tensor to select elements from.
        indices: A symbolic tensor of index values to use for selection.
        axis: The dimension which ``indices`` indexes from ``input``.
            If negative, indexes relative to the end of the input tensor.
            For instance, ``gather(input, indices, axis=-1)`` will index
            against the last dimension of ``input``.

    Returns:
        A new symbolic tensor representing the result of the gather operation.
    """
    input, indices = TensorValue(input), TensorValue(indices)
    shape = input.shape
    output_shape = [*shape[:axis], *indices.shape, *shape[axis + 1 :]]
    return Graph.current._add_op(
        rmo.mo_gather,
        TensorType(input.dtype, output_shape).to_mlir(),
        input,
        indices,
        constant(axis, DType.int64),
    )[0].tensor
