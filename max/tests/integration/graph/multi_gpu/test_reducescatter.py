# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Test the max.engine Python bindings with reducescatter operation."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from max.driver import (
    CPU,
    Accelerator,
    Buffer,
    Device,
    accelerator_count,
)
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.nn import Module, Signals

M = 512
N = 1024


def reducescatter_graph(signals: Signals) -> Graph:
    devices = signals.devices
    num_devices = len(devices)

    # Create input types for each device
    input_types = [
        TensorType(dtype=DType.float32, shape=[M, N], device=devices[i])
        for i in range(num_devices)
    ]
    # Combine tensor types and buffer types
    all_input_types = input_types + list(signals.input_types())

    with Graph(
        "reducescatter",
        input_types=all_input_types,
    ) as graph:
        # Get tensor inputs and apply scaling
        tensor_inputs = []
        for i in range(num_devices):
            assert isinstance(graph.inputs[i], TensorValue)
            # Scale each input by (i + 1)
            scaled_input = graph.inputs[i].tensor * (i + 1)
            tensor_inputs.append(scaled_input)

        reducescatter_outputs = ops.reducescatter.sum(
            tensor_inputs,
            [inp.buffer for inp in graph.inputs[num_devices:]],
        )

        graph.output(*reducescatter_outputs)
        return graph


def test_reducescatter_execution() -> None:
    """Tests multi-device reducescatter execution."""
    # Use available GPUs, minimum 2, maximum 4
    available_gpus = accelerator_count()
    if available_gpus < 2:
        pytest.skip("Test requires at least 2 GPUs")

    num_gpus = min(available_gpus, 4)

    signals = Signals(devices=[DeviceRef.GPU(id=id) for id in range(num_gpus)])
    graph = reducescatter_graph(signals)
    host = CPU()

    # Create device objects
    devices: list[Device]
    devices = [Accelerator(i) for i in range(num_gpus)]

    session = InferenceSession(devices=[host] + devices)
    compiled = session.load(graph)

    # Create input tensors
    a_np = np.ones((M, N)).astype(np.float32)
    # Expected reduced value: sum of (1 * 1) + (1 * 2) + ... + (1 * num_gpus)
    # = 1 + 2 + ... + num_gpus = num_gpus * (num_gpus + 1) / 2
    expected_sum = num_gpus * (num_gpus + 1) // 2

    # Create tensors on each device
    input_tensors = [Buffer.from_numpy(a_np).to(device) for device in devices]

    output = compiled.execute(*input_tensors, *signals.buffers())

    # Check Executed Graph
    # Output shape should be [M, N/num_gpus] for each device
    expected_cols = N // num_gpus
    for out_tensor, device in zip(output, devices, strict=True):
        assert isinstance(out_tensor, Buffer)
        assert out_tensor.device == device
        result = out_tensor.to(host).to_numpy()
        assert result.shape == (M, expected_cols)
        # Each device gets a portion of the reduced result
        expected_out = np.full(
            (M, expected_cols), expected_sum, dtype=np.float32
        )
        assert np.allclose(expected_out, result)


class ReduceScatterAdd(Module):
    """A fused reducescatter with an elementwise add."""

    num_devices: int
    """Number of devices to reducescatter between."""

    def __init__(self, num_devices: int) -> None:
        super().__init__()
        self.num_devices = num_devices

    def __call__(
        self,
        *args: TensorValue | BufferValue,
    ) -> list[TensorValue]:
        # Split args into tensor inputs and signal buffers
        inputs = [cast(TensorValue, arg) for arg in args[: self.num_devices]]
        signal_buffers = [
            cast(BufferValue, arg) for arg in args[self.num_devices :]
        ]

        # Reducescatter
        results = ops.reducescatter.sum(inputs, signal_buffers)

        biases = [
            ops.constant(42, dtype=DType.float32, device=DeviceRef.GPU(id))
            for id in range(self.num_devices)
        ]

        # Elementwise add that should fuse into reducescatter's epilogue.
        return [x + y for x, y in zip(results, biases, strict=True)]


@pytest.mark.parametrize("num_gpus", [2, 4])
def test_reducescatter_epilogue_fusion(num_gpus: int) -> None:
    """Tests that an elementwise add correctly follows a reducescatter operation."""
    if (available_gpus := accelerator_count()) < num_gpus:
        pytest.skip(
            f"skipping {num_gpus=} test since only {available_gpus} available"
        )

    graph_devices = [DeviceRef.GPU(id) for id in range(num_gpus)]
    signals = Signals(devices=graph_devices)

    host = CPU()
    devices: list[Device] = [Accelerator(i) for i in range(num_gpus)]
    session = InferenceSession(devices=[host] + devices)

    model = ReduceScatterAdd(num_devices=len(devices))
    graph = Graph(
        "ReduceScatterAdd_fusion",
        forward=model,
        input_types=[
            *[
                TensorType(DType.float32, shape=[M, N], device=graph_devices[i])
                for i in range(num_gpus)
            ],
            *signals.input_types(),
        ],
    )

    compiled = session.load(graph)

    inputs = []
    a_np = np.ones((M, N), np.float32)
    for i in range(num_gpus):
        inputs.append(Buffer.from_numpy(a_np).to(devices[i]))

    for dev in devices:
        dev.synchronize()

    outputs = compiled.execute(*inputs, *signals.buffers())

    # Expected: sum of all inputs (num_gpus ones) + 42 bias
    # Each input is ones, so sum = num_gpus, plus bias = 42
    expected_cols = N // num_gpus
    expected = np.full((M, expected_cols), num_gpus + 42.0, dtype=np.float32)

    for tensor in outputs:
        assert isinstance(tensor, Buffer)
        result = tensor.to(host).to_numpy()
        assert result.shape == (M, expected_cols)
        assert np.allclose(expected, result, atol=1e-6)
