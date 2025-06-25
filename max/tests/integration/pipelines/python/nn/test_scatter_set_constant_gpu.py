# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the scatter_set_constant kernel."""

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType
from max.nn.kernels import scatter_set_constant


@pytest.mark.parametrize("device", [CPU(), Accelerator()])
def test_scatter_set_constant_2d(device: Device) -> None:
    vocab_size = 3
    data = Tensor.from_numpy(np.zeros((3, 3), dtype=np.float32)).to(device)
    indices = Tensor.from_numpy(
        np.array([[0, 1], [1, 1], [1, 2], [2, 0], [2, 2]], dtype=np.int32)
    ).to(device)
    fill_val = 5.0

    with Graph(
        "scatter_set_constant_graph",
        input_types=[
            BufferType(
                DType.float32,
                ("seq_len", vocab_size),
                device=DeviceRef.from_device(device),
            ),
            TensorType(
                DType.int32,
                ("num_indices", 2),
                device=DeviceRef.from_device(device),
            ),
        ],
    ) as graph:
        data_val = graph.inputs[0].buffer
        indices_val = graph.inputs[1].tensor
        scatter_set_constant(data_val, indices_val, fill_val)
        graph.output(data_val)

    session = InferenceSession(devices=[device])
    model = session.load(graph)

    model(data, indices)

    expected_output = np.array(
        [
            [
                0,
                5,
                0,
            ],
            [
                0,
                5,
                5,
            ],
            [
                5,
                0,
                5,
            ],
        ]
    ).astype(np.float32)
    np.testing.assert_array_equal(data.to_numpy(), expected_output)
