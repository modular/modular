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
) -> None:
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
        out = ops.scatter(input_val, updates, indices, axis)
        graph.output(out)

    model = session.load(graph)
    input_tensor = Tensor.from_numpy(input).to(model.input_devices[0])

    result = model.execute(input_tensor)[0]
    assert isinstance(result, Tensor)

    np.testing.assert_equal(result.to_numpy(), np.float32(expected))


@pytest.mark.parametrize(
    "input_data,updates_data,indices_data,expected",
    [
        # 1D scatter_nd
        (
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 20.0],
            [[1], [3]],
            [1.0, 10.0, 3.0, 20.0, 5.0],
        ),
        # 2D scatter_nd with 1D indices
        (
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]],
            [[0], [2]],
            [[10.0, 11.0, 12.0], [4.0, 5.0, 6.0], [13.0, 14.0, 15.0]],
        ),
        # 2D scatter_nd with 2D indices
        (
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [10.0, 20.0, 30.0],
            [[0, 1], [1, 2], [2, 0]],
            [[1.0, 10.0, 3.0], [4.0, 5.0, 20.0], [30.0, 8.0, 9.0]],
        ),
        # Empty updates
        (
            [1.0, 2.0, 3.0, 4.0],
            [],
            np.empty((0, 1), dtype=np.int32),
            [1.0, 2.0, 3.0, 4.0],
        ),
    ],
)
def test_scatter_nd(
    session: InferenceSession, input_data, updates_data, indices_data, expected
) -> None:
    """Test scatter_nd operation with various input configurations."""
    input_array = np.array(input_data, dtype=np.float32)
    input_type = TensorType(DType.float32, input_array.shape, device_ref)

    with Graph("scatter_nd", input_types=[input_type]) as graph:
        input_val = graph.inputs[0].tensor
        updates = ops.constant(
            np.array(updates_data, dtype=np.float32),
            DType.float32,
            device=device_ref,
        )
        indices = ops.constant(
            np.array(indices_data, dtype=np.int32),
            DType.int32,
            device=device_ref,
        )
        out = ops.scatter_nd(input_val, updates, indices)
        graph.output(out)

    model = session.load(graph)
    input_tensor = Tensor.from_numpy(input_array).to(model.input_devices[0])

    result = model.execute(input_tensor)[0]
    assert isinstance(result, Tensor)

    np.testing.assert_equal(result.to_numpy(), np.float32(expected))
