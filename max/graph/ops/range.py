# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for range."""

from __future__ import annotations

from max.mlir.dialects import rmo

from ..graph import DeviceRef, Graph
from ..type import DimLike
from ..value import TensorType, TensorValue, TensorValueLike


def range(
    start: TensorValueLike,
    stop: TensorValueLike,
    step: TensorValueLike,
    out_dim: DimLike,
    device: DeviceRef | None = None,
) -> TensorValue:
    """Creates a sequence of numbers. The sequence goes from `start` with
    increments of size `step` up to (but not including) `stop`. All arguments
    are mandatory and must have the same element type.

    Note the following restrictions on input values:
    1. `step` must be non-zero
    2. `stop - start` must be zero or have the same sign as `step`

    Args:
        start: The start of the range to generate.
        stop: The range will be generated up to, but not including, this value.
        step: The step size for the range.
        out_dim: The expected output dimensions returned by the range op.
          These will be assert at graph execution time to be correct.
        device: Device of the result tensor.

    Returns:
        A symbolic tensor value containing the defined range of values.
    """
    start = TensorValue(start)
    stop = TensorValue(stop)
    step = TensorValue(step)

    if start.dtype != stop.dtype or stop.dtype != step.dtype:
        raise ValueError("range expected inputs of the same type!")
    if start.rank != 0 or stop.rank != 0 or step.rank != 0:
        raise ValueError("range expected scalar values as inputs!")

    return Graph.current._add_op(
        rmo.mo_range,
        TensorType(start.dtype, shape=[out_dim], device=device).to_mlir(),
        start,
        stop,
        step,
    )[0].tensor
