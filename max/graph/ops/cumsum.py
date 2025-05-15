# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for cumsum."""

from max import mlir
from max.dtype import DType
from max.mlir.dialects import rmo

from ..graph import Graph
from ..type import DeviceRef
from ..value import TensorValue, TensorValueLike
from .constant import constant


def cumsum(
    x: TensorValueLike,
    axis: int = -1,
    exclusive: bool = False,
    reverse: bool = False,
) -> TensorValue:
    """Computes the cumulative sum of the input tensor along the given axis.

    Args:
        x: The input tensor to sum over.
        axis: The axis along which to compute the sum. If negative,
            indexes from the last dimension. For example, a value of -1 will
            compute the sum along the last dimension.
        exclusive: If set, start at 0 and exclude the final element.
            Otherwise, start with the first element. Said another way, cumsum
            computes `[sum(x[..., :i, ...]) for i in range(x.shape[axis])]`.
            If exclusive is set, the bounds are instead `range(1, x.shape[axis])`.
        reverse: If set, start from the end. In other words, the first element
            will be the total sum, with each element following counting
            downwards; or `[sum(x[..., i:, ...]) for i in range(x.shape[axis])]`.

    Returns:
        A symbolic tensor representing the result of the cumsum operation.
        The tensor will have the same type as the input tensor. The computed
        values will be the cumulative sum of the values along the given axis,
        according to the specified parameters:

        - if `exclusive` is set, the first value will be 0, and the last
          value will be excluded from the sum
        - if `reverse` is set, the sum will be computed starting at the
          back of the axis back to the front, rather than front-to-back
    """
    x = TensorValue(x)

    if axis < 0:
        axis += x.rank
    if not 0 <= axis < x.rank:
        raise ValueError(f"Invalid {axis=} for input {x.rank=}")

    # TODO(KERN-1095): Add support for GPU kernel for cumsum and remove manual transfers
    original_device = x.type.device
    x = x.to(DeviceRef.CPU())
    answer = Graph.current._add_op(
        rmo.mo_cumsum,
        x.type.to_mlir(),
        x,
        constant(axis, DType.int64, DeviceRef.CPU()),
        exclusive=mlir.IntegerAttr.get(mlir.IndexType.get(), int(exclusive)),
        reverse=mlir.IntegerAttr.get(mlir.IndexType.get(), int(reverse)),
    )[0].tensor
    return answer.to(original_device)
