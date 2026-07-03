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

"""Tests for bundled allreduce (mo.parallel + mo.bundled.allreduce.sum)."""

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
from max.nn import Signals

M = 512
N = 1024


def bundled_allreduce_graph(signals: Signals) -> Graph:
    devices = signals.devices
    num_devices = len(devices)

    input_types = [
        TensorType(dtype=DType.float32, shape=[M, N], device=devices[i])
        for i in range(num_devices)
    ]
    all_input_types = input_types + list(signals.input_types())

    with Graph(
        "bundled_allreduce",
        input_types=all_input_types,
    ) as graph:
        tensor_inputs = []
        for i in range(num_devices):
            assert isinstance(graph.inputs[i], TensorValue)
            scaled_input = graph.inputs[i].tensor * (i + 1)
            tensor_inputs.append(scaled_input)

        signal_buffers = [
            cast(BufferValue, graph.inputs[num_devices + i])
            for i in range(num_devices)
        ]

        allreduce_outputs = ops.bundled_allreduce.sum(
            tensor_inputs, signal_buffers
        )

        graph.output(*allreduce_outputs)
        return graph


def test_bundled_allreduce_execution() -> None:
    """Tests multi-device bundled allreduce execution."""
    available_gpus = accelerator_count()
    if available_gpus < 2:
        pytest.skip("Test requires at least 2 GPUs")

    num_gpus = min(available_gpus, 4)

    signals = Signals(devices=[DeviceRef.GPU(id=id) for id in range(num_gpus)])
    graph = bundled_allreduce_graph(signals)
    host = CPU()

    devices: list[Device]
    devices = [Accelerator(i) for i in range(num_gpus)]

    session = InferenceSession(devices=[host] + devices)
    compiled = session.load(graph)

    a_np = np.ones((M, N)).astype(np.float32)
    # Expected: sum of (1*1) + (1*2) + ... + (1*num_gpus)
    expected_sum = num_gpus * (num_gpus + 1) // 2
    out_np = a_np * expected_sum

    input_tensors = [Buffer.from_numpy(a_np).to(device) for device in devices]

    output = compiled.execute(*input_tensors, *signals.buffers())

    for out_tensor, device in zip(output, devices, strict=True):
        assert isinstance(out_tensor, Buffer)
        assert out_tensor.device == device
        assert np.allclose(out_np, out_tensor.to(host).to_numpy())


@pytest.mark.parametrize("num_gpus", [2, 4])
def test_bundled_allreduce_relu_fusion(num_gpus: int) -> None:
    """Tests bundled allreduce followed by relu.

    Negative inputs make the allreduce sum negative; relu clips to zero,
    confirming it is applied.
    """
    if (available_gpus := accelerator_count()) < num_gpus:
        pytest.skip(
            f"skipping {num_gpus=} test since only {available_gpus} available"
        )

    graph_devices = [DeviceRef.GPU(id) for id in range(num_gpus)]
    signals = Signals(devices=graph_devices)

    input_types = [
        TensorType(DType.float32, shape=[M, N], device=graph_devices[i])
        for i in range(num_gpus)
    ]
    all_input_types = list(input_types) + list(signals.input_types())

    with Graph(
        "bundled_allreduce_relu",
        input_types=all_input_types,
    ) as graph:
        tensor_inputs = [
            cast(TensorValue, graph.inputs[i]) for i in range(num_gpus)
        ]
        signal_buffers = [
            cast(BufferValue, graph.inputs[num_gpus + i])
            for i in range(num_gpus)
        ]

        allreduce_outputs = ops.bundled_allreduce.sum(
            tensor_inputs, signal_buffers
        )
        relu_outputs = [ops.relu(t) for t in allreduce_outputs]

        graph.output(*relu_outputs)

    host = CPU()
    devices: list[Device] = [Accelerator(i) for i in range(num_gpus)]
    session = InferenceSession(devices=[host] + devices)
    compiled = session.load(graph)

    # Feed negative-ones so allreduce sum = -num_gpus per element.
    # relu(-num_gpus) = 0, confirming the relu is not dropped.
    a_np = -np.ones((M, N), np.float32)
    input_tensors = [Buffer.from_numpy(a_np).to(dev) for dev in devices]
    for dev in devices:
        dev.synchronize()

    outputs = compiled.execute(*input_tensors, *signals.buffers())

    expected = np.zeros((M, N), dtype=np.float32)
    for tensor in outputs:
        assert isinstance(tensor, Buffer)
        result_np = tensor.to(host).to_numpy()
        assert np.allclose(expected, result_np, atol=1e-6), (
            f"Expected all zeros (relu of negative sum) but got "
            f"min={result_np.min()}, max={result_np.max()}"
        )
