# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for stack."""

from collections.abc import Iterable

from ..value import TensorValue, TensorValueLike
from .concat import concat
from .unsqueeze import unsqueeze
from .utils import check_axis_in_bounds


def _axis_bounds(rank: int) -> tuple[int, int]:
    # For stack, valid axis range is [-rank+1, rank] because we're inserting a new dimension
    return -(rank + 1), rank


def _axis_out_of_range_error(
    axis: int, lower_bound: int, upper_bound: int
) -> str:
    return f"Axis out of range (expected to be in range of [{lower_bound}, {upper_bound}], but got {axis})"


def _stack_axis_bounds(rank: int) -> tuple[int, int]:
    # For stack, valid axis range is [-rank-1, rank] because we're inserting a new dimension
    return -(rank + 1), rank


def _check_stack_axis_in_bounds(axis: int, rank: int) -> None:
    lower_bound, upper_bound = _stack_axis_bounds(rank)
    if axis < lower_bound or axis > upper_bound:
        raise IndexError(
            _axis_out_of_range_error(axis, lower_bound, upper_bound)
        )


def stack(values: Iterable[TensorValueLike], axis: int = 0) -> TensorValue:
    """Stacks a list of tensors along a new axis.

    Args:
        values: A list of symbolic tensor values. Each tensor must have the same
            dtype and rank, and must have the same dimension size for each
            dimension.
        axis: The axis to concatenate along. If negative, indexes relative
            to the end of the tensor shape *plus 1*. For instance,
            ``stack(vs, -1)`` will create and stack along a new axis as the
            last dimension, aad ``stack(vs, -2)`` will create and stack along a new
            dimension which is inserted immediately before the last dimension.

    Returns:
        A new symbolic tensor representing the result of the stack. It will
        have rank ``n+1`` where ``n`` is the rank of each input tensor. Its size
        on each dimension other than ``axis`` will be the same as each input tensors',
        with the new axis inserted. Along the new dimension it will have size
        ``len(values)``.
    """
    values_coerced = [TensorValue(v) for v in values]
    if len(values_coerced) == 0:
        raise ValueError("Expected at least one value to stack")

    rank = len(values_coerced[0].shape)
    if any(len(v.shape) != rank for v in values_coerced):
        raise ValueError("All inputs to stack must be the same rank")

    if any(v.dtype != values_coerced[0].dtype for v in values_coerced):
        raise ValueError("All inputs to stack must have the same dtype")

    if any(v.device != values_coerced[0].device for v in values_coerced):
        raise ValueError("All inputs to stack must have the same device")

    # Check if axis is within bounds
    check_axis_in_bounds(axis, rank, _axis_bounds)

    unsqueezed = [unsqueeze(v, axis) for v in values_coerced]

    # Short circuit to avoid bloating graph with unneeded op.
    if len(unsqueezed) == 1:
        return unsqueezed[0]

    return concat(unsqueezed, axis=axis)
