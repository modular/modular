# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Test the max.engine Python bindings with Max Graph when using explicit device."""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
)

M = 512
N = 1024


def allgather_graph(devices, dim) -> Graph:
    with Graph(
        "allgather",
        input_types=[
            TensorType(dtype=DType.float32, shape=[M, N], device=device)
            for device in devices
        ],
    ) as graph:
        for input in graph.inputs:
            assert isinstance(input, TensorValue)
        allgather_outputs = ops.allgather(
            (v.tensor for v in graph.inputs), dim=dim
        )
        graph.output(*allgather_outputs)
        return graph


@pytest.mark.parametrize("num_gpus, dim", [(1, 0), (2, 0), (4, 0)])
def test_allgather_execution(num_gpus: int, dim: int) -> None:
    """Tests multi-device allgather execution."""

    if num_gpus > accelerator_count():
        pytest.skip(
            f"Not enough GPUs to run allgather test with {num_gpus} GPUs."
        )
    graph = allgather_graph(
        [DeviceRef.GPU(id=id) for id in range(num_gpus)], dim
    )
    host = CPU()
    devices = [Accelerator(n) for n in range(num_gpus)]

    session = InferenceSession(devices=[host, *devices])
    compiled = session.load(graph)

    # Set up input values so that the gathered output is a range from
    # 0 to the total number of elements in all inputs.
    numpy_inputs = []
    tensor_inputs = []
    stride = M * N
    numpy_inputs = [
        np.arange(stride).reshape((M, N)) + (stride * i)
        for i in range(num_gpus)
    ]
    tensor_inputs = [
        Tensor.from_numpy(a.astype(np.float32)).to(device)
        for a, device in zip(numpy_inputs, devices)
    ]

    # Synchronize devices so that the signal buffers are initialized.
    for dev in devices:
        dev.synchronize()

    outputs = compiled.execute(*tensor_inputs)

    expected_output = np.concatenate(numpy_inputs, axis=dim)

    for n, output in enumerate(outputs):
        assert isinstance(output, Tensor)
        assert output.device == devices[n]
        assert np.equal(output.to(host).to_numpy(), expected_output).all()
