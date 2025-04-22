# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

import numpy as np
import pytest
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, ops


@pytest.mark.parametrize(
    "input,updates,indices,axis,expected",
    [
        (
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            [[1.1, 2.2], [3.3, 4.4]],
            [[0, 1], [3, 2]],
            0,
            [[1.1, 2.0], [3.0, 2.2], [5.0, 4.4], [3.3, 8.0]],
        ),
        (
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            [[1.1, 2.2], [3.3, 4.4]],
            [[1, 0], [0, 1]],
            1,
            [[2.2, 1.1], [3.3, 4.4], [5.0, 6.0], [7.0, 8.0]],
        ),
    ],
)
def test_scatter(
    session: InferenceSession, input, updates, indices, axis, expected
):
    with Graph("scatter", input_types=[]) as graph:
        input = ops.constant(np.array(input), DType.float32)
        updates = ops.constant(np.array(updates), DType.float32)
        indices = ops.constant(np.array(indices), DType.int32)
        axis = ops.constant(axis, DType.int32)
        out = ops.scatter(input, updates, indices, axis)
        graph.output(out)

    model = session.load(graph)
    result = model.execute()[0]
    np.testing.assert_equal(result.to_numpy(), np.float32(expected))
