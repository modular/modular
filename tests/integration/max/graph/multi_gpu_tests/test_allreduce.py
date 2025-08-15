# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

"""Test the max.engine Python bindings with Max Graph when using explicit device."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from max.driver import (
    CPU,
    Accelerator,
    Device,
    Tensor,
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
from max.nn import Allreduce, Module, Signals

M = 512
N = 1024


def allreduce_graph(signals: Signals) -> Graph:
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
        "allreduce",
        input_types=all_input_types,
    ) as graph:
        # Get tensor inputs and apply scaling
        tensor_inputs = []
        for i in range(num_devices):
            assert isinstance(graph.inputs[i], TensorValue)
            # Scale each input by (i + 1)
            scaled_input = graph.inputs[i].tensor * (i + 1)
            tensor_inputs.append(scaled_input)

        allreduce = Allreduce(num_accelerators=num_devices)
        allreduce_outputs = allreduce(
            tensor_inputs,
            [inp.buffer for inp in graph.inputs[num_devices:]],
        )

        graph.output(*allreduce_outputs)
        return graph


def test_allreduce_execution() -> None:
    """Tests multi-device allreduce execution."""
    # Use available GPUs, minimum 2, maximum 4
    available_gpus = accelerator_count()
    if available_gpus < 2:
        pytest.skip("Test requires at least 2 GPUs")

    num_gpus = min(available_gpus, 4)

    signals = Signals(devices=[DeviceRef.GPU(id=id) for id in range(num_gpus)])
    graph = allreduce_graph(signals)
    host = CPU()

    # Create device objects
    devices: list[Device]
    devices = [Accelerator(i) for i in range(num_gpus)]

    session = InferenceSession(devices=[host] + devices)
    compiled = session.load(graph)

    # Create input tensors
    a_np = np.ones((M, N)).astype(np.float32)
    # Expected output: sum of (1 * 1) + (1 * 2) + ... + (1 * num_gpus)
    # = 1 + 2 + ... + num_gpus = num_gpus * (num_gpus + 1) / 2
    expected_sum = num_gpus * (num_gpus + 1) // 2
    out_np = a_np * expected_sum

    # Create tensors on each device
    input_tensors = [Tensor.from_numpy(a_np).to(device) for device in devices]

    # Synchronize devices so that the signal buffers are initialized.
    for dev in devices:
        dev.synchronize()

    output = compiled.execute(*input_tensors, *signals.buffers())

    # Check Executed Graph
    for out_tensor, device in zip(output, devices):
        assert isinstance(out_tensor, Tensor)
        assert out_tensor.device == device
        assert np.allclose(out_np, out_tensor.to(host).to_numpy())


class AllreduceAdd(Module):
    """A fused allreduce with an elementwise add."""

    allreduce: Allreduce
    """Allreduce layer."""

    num_devices: int
    """Number of devices to allreduce between."""

    def __init__(self, num_devices: int) -> None:
        super().__init__()

        self.allreduce = Allreduce(num_accelerators=num_devices)
        self.num_devices = num_devices

    def __call__(
        self,
        *args: TensorValue | BufferValue,
    ) -> list[TensorValue]:
        # Split args into tensor inputs and signal buffers
        # The number of tensor inputs should match the number of devices
        inputs = [cast(TensorValue, arg) for arg in args[: self.num_devices]]
        signal_buffers = [
            cast(BufferValue, arg) for arg in args[self.num_devices :]
        ]

        # Fused Mojo kernel allreduce implementation.
        results = self.allreduce(inputs, signal_buffers)

        biases = [
            ops.constant(42, dtype=DType.float32, device=DeviceRef.GPU(id))
            for id in range(self.num_devices)
        ]

        # Elementwise add that should fuse into allreduce's epilogue.
        return [x + y for x, y in zip(results, biases)]


@pytest.mark.parametrize("num_gpus", [1, 2, 4])
def test_allreduce_epilogue_fusion(num_gpus: int) -> None:
    """Tests that an elementwise add correctly follows an allreduce operation."""
    if (available_gpus := accelerator_count()) < num_gpus:
        pytest.skip(
            f"skipping {num_gpus=} test since only {available_gpus} available"
        )

    graph_devices = [DeviceRef.GPU(id) for id in range(num_gpus)]
    signals = Signals(devices=graph_devices)

    host = CPU()
    devices: list[Device] = [Accelerator(i) for i in range(num_gpus)]
    session = InferenceSession(devices=[host] + devices)

    model = AllreduceAdd(num_devices=len(devices))
    graph = Graph(
        "AllreduceAdd_fusion",
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
        inputs.append(Tensor.from_numpy(a_np).to(devices[i]))

    for dev in devices:
        dev.synchronize()

    outputs = compiled.execute(*inputs, *signals.buffers())

    expected = np.full((M, N), num_gpus + 42.0, dtype=np.float32)

    for tensor in outputs:
        assert isinstance(tensor, Tensor)
        assert np.allclose(expected, tensor.to(host).to_numpy(), atol=1e-6)


def test_allreduce_signal_buffer_too_small_error_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that allreduce provides helpful error message when signal buffer is too small."""
    # Need at least 2 GPUs for actual allreduce communication
    available_gpus = accelerator_count()
    if available_gpus < 2:
        pytest.skip("Test requires at least 2 GPUs")

    # Use 2 GPUs.
    num_gpus = 2

    # Monkeypatch the signal buffer size to be extremely small.
    monkeypatch.setattr(Signals, "NUM_BYTES", 1)

    # Set up multiple GPUs.
    gpu_devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    signals = Signals(devices=gpu_devices)

    # Build graph with inputs on multiple GPUs.
    input_types = [
        TensorType(dtype=DType.float32, shape=[64, 64], device=gpu_devices[i])
        for i in range(num_gpus)
    ]

    # Combine tensor types and signal buffer types.
    all_input_types = input_types + list(signals.input_types())

    with Graph(
        "allreduce_small_signal",
        input_types=all_input_types,
    ) as graph:
        # Get tensor inputs from each GPU
        tensor_inputs = [graph.inputs[i].tensor for i in range(num_gpus)]
        signal_buffers = [inp.buffer for inp in graph.inputs[num_gpus:]]

        # Perform allreduce across GPUs
        allreduce_outputs = ops.allreduce.sum(tensor_inputs, signal_buffers)
        graph.output(*allreduce_outputs)

    # Create session and compile.
    host = CPU()
    devices: list[Device] = [Accelerator(i) for i in range(num_gpus)]
    session = InferenceSession(devices=[host] + devices)
    compiled = session.load(graph)

    # Create input tensors on each device.
    input_tensors = [
        Tensor.zeros((64, 64), dtype=DType.float32).to(devices[i])
        for i in range(num_gpus)
    ]

    # Synchronize devices.
    for dev in devices:
        dev.synchronize()

    # Execute and expect error.
    error_regex = r"Expected signal buffer to be at least \d+ bytes, but got \d+\. This error can appear when running large requests through MAX serve without chunked prefill\. If so, try enabling chunked prefill with --enable-chunked-prefill\. Otherwise, consider increasing the signal buffer size\."

    with pytest.raises(ValueError, match=error_regex):
        compiled.execute(*input_tensors, *signals.buffers())
