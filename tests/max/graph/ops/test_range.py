# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""ops.range tests."""

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, ops


@pytest.mark.parametrize(
    "start, stop, step",
    [
        (0, 5, 1),  # normal positive step, start < stop
        (5, 0, -1),  # normal negative step, start > stop
        (0, 0, 1),  # start == stop (empty range)
        (-5, 0, 1),  # negative start, positive step
        (0, -5, -1),  # negative step, start > stop
        (0, 5, 2),  # step does not evenly divide into stop - start
        (0, 5, 10),  # step larger than stop - start
        (0, 2**63 - 1, 1),  # full int64 range
        (-1, -(2**63), -1),  # full int64 range, negative step
    ],
)
def test_range(start: int, stop: int, step: int) -> None:
    """Tests ops.range cases that should pass."""
    with Graph("range", input_types=()) as graph:
        dim = (stop - start) // step
        start_val = ops.constant(start, DType.int64)
        stop_val = ops.constant(stop, DType.int64)
        step_val = ops.constant(step, DType.int64)
        out = ops.range(
            start_val, stop_val, step_val, dim, device=DeviceRef.CPU()
        )
        graph.output(out)


@pytest.mark.parametrize(
    "start, stop, step",
    [
        # TODO(bduke): step == 0 should raise in the range op builder.
        # (0, 5, 0),  # step = 0
        (0, 2**63, 1),  # dim exceeds int64 max
        (0, -(2**63), -1),  # dim exceeds int64 max in negative direction
        (2**62, 2**63, 1),  # large range, dim exceeds int64 max
        (-(2**63), 0, 1),  # large negative start
    ],
)
def test_range_exceptions(start: int, stop: int, step: int) -> None:
    """Tests ops.range cases that should raise an exception."""
    with pytest.raises(ValueError):
        with Graph("range", input_types=()) as graph:
            # Set dim to 0 as a placeholder when we would divide by zero.
            dim = (stop - start) // step if step != 0 else 0
            start_val = ops.constant(start, DType.int64)
            stop_val = ops.constant(stop, DType.int64)
            step_val = ops.constant(step, DType.int64)
            out = ops.range(
                start_val, stop_val, step_val, dim, device=DeviceRef.CPU()
            )
            graph.output(out)
