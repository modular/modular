# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for transpose."""

import numpy as np
from max.dtype import DType
from max.mlir.dialects import rmo

from ..graph import Graph
from ..type import DeviceRef
from ..value import TensorType, TensorValue, TensorValueLike
from .constant import constant


def _axis_out_of_range_error(
    axis_name: str, axis: int, lower_bound: int, upper_bound: int
) -> str:
    return f"Axis {axis_name} out of range (expected to be in range of [{lower_bound}, {upper_bound}], but got {axis})"


def _axis_bounds(rank: int) -> tuple[int, int]:
    if rank == 0:
        return -1, 0
    return -rank, rank - 1


def _check_axis_in_bounds(axis: int, axis_name: str, rank: int) -> None:
    lower_bound, upper_bound = _axis_bounds(rank)
    if axis < lower_bound or axis > upper_bound:
        raise IndexError(
            _axis_out_of_range_error(axis_name, axis, lower_bound, upper_bound)
        )


def transpose(x: TensorValueLike, axis_1: int, axis_2: int) -> TensorValue:
    """Transposes two axes of a symbolic tensor.
    For more information, see :obj:`~max.graph.TensorValue.transpose()`.

    Args:
        x: The input symbolic tensor to transpose.
        axis_1: One of the two axes to transpose. If negative, this indexes
           from the end of the tensor. For example,
           :code:`transpose(v, -1, -2)` transposes the last two axes.
        axis_2: The other axis to transpose. May also be negative to index from
           the end of the tensor.

    Returns:
        A new symbolic tensor with the two specified axes transposed.
        It has the same elements and dtype, but the order of the elements
        is different according to the transposition.
    """
    v = TensorValue(x)

    rank = len(v.shape)

    _check_axis_in_bounds(axis_1, "axis_1", rank)
    _check_axis_in_bounds(axis_2, "axis_2", rank)

    if axis_1 < 0:
        axis_1 += rank
    if axis_2 < 0:
        axis_2 += rank

    new_shape = v.shape
    indices = np.array(range(len(new_shape)))

    # Only change the shape for non-zero rank tensors.
    if rank > 0:
        new_shape[axis_1], new_shape[axis_2] = (
            new_shape[axis_2],
            new_shape[axis_1],
        )
        indices[axis_1], indices[axis_2] = axis_2, axis_1

    return Graph.current._add_op(
        rmo.mo_transpose,
        TensorType(dtype=v.dtype, shape=new_shape, device=v.device).to_mlir(),
        v,
        constant(indices, DType.int64, DeviceRef.CPU()),
    )[0].tensor
