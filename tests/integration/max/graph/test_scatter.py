# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

import numpy as np
import pytest
from max.driver import Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

device_ref = DeviceRef.GPU() if accelerator_count() > 0 else DeviceRef.CPU()


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
    input = np.array(input, dtype=np.float32)
    input_type = TensorType(DType.float32, input.shape, device_ref)
    with Graph("scatter", input_types=[input_type]) as graph:
        input_val = graph.inputs[0].tensor
        updates = ops.constant(
            np.array(updates), DType.float32, device=device_ref
        )
        indices = ops.constant(
            np.array(indices), DType.int32, device=device_ref
        )
        axis = ops.constant(axis, DType.int32, device=DeviceRef.CPU())
        out = ops.scatter(input_val, updates, indices, axis)
        graph.output(out)

    model = session.load(graph)
    input_tensor = Tensor.from_numpy(input).to(model.input_devices[0])

    result = model.execute(input_tensor)[0]
    assert isinstance(result, Tensor)

    np.testing.assert_equal(result.to_numpy(), np.float32(expected))
