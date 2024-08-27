# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for gather."""


from max.dtype import DType
from max.mlir.dialects import rmo

from ..graph import Graph
from ..value import TensorValue, ValueLike
from ..type import TensorType
from .constant import scalar


def gather(input: ValueLike, indices: ValueLike, axis: int = -1) -> TensorValue:
    input, indices = TensorValue(input), TensorValue(indices)
    shape = input.shape
    output_shape = [*shape[:axis], *indices.shape, *shape[axis + 1 :]]
    return Graph.current._add_op(
        rmo.mo_gather,
        TensorType(input.dtype, output_shape).to_mlir(),
        input,
        indices,
        scalar(axis, DType.int64),
    )[0].tensor
