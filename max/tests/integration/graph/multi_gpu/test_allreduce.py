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

"""Test the max.engine Python bindings with Max Graph when using explicit device."""

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
from max.nn.legacy import Allreduce, Module, Signals

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
    input_tensors = [Buffer.from_numpy(a_np).to(device) for device in devices]

    output = compiled.execute(*input_tensors, *signals.buffers())

    # Check Executed Graph
    for out_tensor, device in zip(output, devices, strict=True):
        assert isinstance(out_tensor, Buffer)
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
        return [x + y for x, y in zip(results, biases, strict=True)]


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
        inputs.append(Buffer.from_numpy(a_np).to(devices[i]))

    for dev in devices:
        dev.synchronize()

    outputs = compiled.execute(*inputs, *signals.buffers())

    expected = np.full((M, N), num_gpus + 42.0, dtype=np.float32)

    for tensor in outputs:
        assert isinstance(tensor, Buffer)
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
        Buffer.zeros((64, 64), dtype=DType.float32).to(devices[i])
        for i in range(num_gpus)
    ]

    # Execute and expect error.
    error_regex = r"Expected signal buffer to be at least \d+ bytes, but got \d+\. This error can appear when running large requests through MAX serve without chunked prefill\. If so, try enabling chunked prefill with --enable-chunked-prefill\. Otherwise, consider increasing the signal buffer size\."

    with pytest.raises(ValueError, match=error_regex):
        compiled.execute(*input_tensors, *signals.buffers())


class DoubleAllreduceWithOp(Module):
    """Two allreduce operations with an intermediate element-wise multiply.

    Tests the pattern: allreduce -> other_kernel -> allreduce.
    """

    allreduce1: Allreduce
    allreduce2: Allreduce
    num_devices: int

    def __init__(self, num_devices: int) -> None:
        super().__init__()
        self.allreduce1 = Allreduce(num_accelerators=num_devices)
        self.allreduce2 = Allreduce(num_accelerators=num_devices)
        self.num_devices = num_devices

    def __call__(
        self,
        *args: TensorValue | BufferValue,
    ) -> list[TensorValue]:
        inputs = [cast(TensorValue, arg) for arg in args[: self.num_devices]]
        signal_buffers = [
            cast(BufferValue, arg) for arg in args[self.num_devices :]
        ]

        first_results = self.allreduce1(inputs, signal_buffers)
        intermediate_results = [x * 2.0 for x in first_results]
        second_results = self.allreduce2(intermediate_results, signal_buffers)

        return second_results


@pytest.mark.parametrize("num_gpus", [2, 4, 8])
def test_allreduce_chained_with_intermediate_op(num_gpus: int) -> None:
    """Tests allreduce -> other_kernel -> allreduce."""
    if (available_gpus := accelerator_count()) < num_gpus:
        pytest.skip(
            f"skipping {num_gpus=} test since only {available_gpus} available"
        )

    graph_devices = [DeviceRef.GPU(id) for id in range(num_gpus)]
    signals = Signals(devices=graph_devices)

    host = CPU()
    devices: list[Device] = [Accelerator(i) for i in range(num_gpus)]
    session = InferenceSession(devices=[host] + devices)

    model = DoubleAllreduceWithOp(num_devices=num_gpus)
    graph = Graph(
        "DoubleAllreduce",
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

    # First allreduce sums to num_gpus, multiply by 2 gives 2*num_gpus,
    # second allreduce sums num_gpus copies: 2*num_gpus^2
    expected = np.full((M, N), 2.0 * num_gpus * num_gpus, dtype=np.float32)

    for tensor in outputs:
        assert isinstance(tensor, Buffer)
        assert np.allclose(expected, tensor.to(host).to_numpy(), atol=1e-6)


class AllreduceFollowedBySubgraph(Module):
    """Allreduce followed by a sequence of fused operations.

    Tests the pattern: allreduce -> subgraph (multiply, add, relu).
    """

    allreduce: Allreduce
    num_devices: int

    def __init__(self, num_devices: int) -> None:
        super().__init__()
        self.allreduce = Allreduce(num_accelerators=num_devices)
        self.num_devices = num_devices

    def __call__(
        self,
        *args: TensorValue | BufferValue,
    ) -> list[TensorValue]:
        inputs = [cast(TensorValue, arg) for arg in args[: self.num_devices]]
        signal_buffers = [
            cast(BufferValue, arg) for arg in args[self.num_devices :]
        ]

        allreduce_results = self.allreduce(inputs, signal_buffers)

        subgraph_results = []
        for result in allreduce_results:
            x = result * 3.0
            x = x + 10.0
            x = ops.relu(x)
            subgraph_results.append(x)

        return subgraph_results


@pytest.mark.parametrize("num_gpus", [2, 4, 8])
def test_allreduce_followed_by_subgraph(num_gpus: int) -> None:
    """Tests allreduce -> subgraph (multiply, add, relu)."""
    if (available_gpus := accelerator_count()) < num_gpus:
        pytest.skip(
            f"skipping {num_gpus=} test since only {available_gpus} available"
        )

    graph_devices = [DeviceRef.GPU(id) for id in range(num_gpus)]
    signals = Signals(devices=graph_devices)

    host = CPU()
    devices: list[Device] = [Accelerator(i) for i in range(num_gpus)]
    session = InferenceSession(devices=[host] + devices)

    model = AllreduceFollowedBySubgraph(num_devices=num_gpus)
    graph = Graph(
        "AllreduceSubgraph",
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

    # allreduce sums to num_gpus, * 3 + 10, relu is no-op
    expected = np.full((M, N), num_gpus * 3.0 + 10.0, dtype=np.float32)

    for tensor in outputs:
        assert isinstance(tensor, Buffer)
        assert np.allclose(expected, tensor.to(host).to_numpy(), atol=1e-6)


class SubgraphFollowedByAllreduce(Module):
    """Sequence of fused operations followed by allreduce.

    Tests the pattern: subgraph (scale, add, max) -> allreduce.
    """

    allreduce: Allreduce
    num_devices: int

    def __init__(self, num_devices: int) -> None:
        super().__init__()
        self.allreduce = Allreduce(num_accelerators=num_devices)
        self.num_devices = num_devices

    def __call__(
        self,
        *args: TensorValue | BufferValue,
    ) -> list[TensorValue]:
        inputs = [cast(TensorValue, arg) for arg in args[: self.num_devices]]
        signal_buffers = [
            cast(BufferValue, arg) for arg in args[self.num_devices :]
        ]

        subgraph_results = []
        for i, inp in enumerate(inputs):
            x = inp * float(i + 1)
            x = x + 5.0
            x = ops.max(x, 0.0)
            subgraph_results.append(x)

        allreduce_results = self.allreduce(subgraph_results, signal_buffers)

        return allreduce_results


@pytest.mark.parametrize("num_gpus", [2, 4, 8])
def test_subgraph_followed_by_allreduce(num_gpus: int) -> None:
    """Tests subgraph (scale, add, max) -> allreduce."""
    if (available_gpus := accelerator_count()) < num_gpus:
        pytest.skip(
            f"skipping {num_gpus=} test since only {available_gpus} available"
        )

    graph_devices = [DeviceRef.GPU(id) for id in range(num_gpus)]
    signals = Signals(devices=graph_devices)

    host = CPU()
    devices: list[Device] = [Accelerator(i) for i in range(num_gpus)]
    session = InferenceSession(devices=[host] + devices)

    model = SubgraphFollowedByAllreduce(num_devices=num_gpus)
    graph = Graph(
        "SubgraphAllreduce",
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

    # Device i computes max(1*(i+1) + 5, 0) = i+6 (0-indexed: i+1+5)
    # sum = num_gpus*(num_gpus+1)/2 + 5*num_gpus
    expected_sum = (num_gpus * (num_gpus + 1)) // 2 + 5 * num_gpus
    expected = np.full((M, N), float(expected_sum), dtype=np.float32)

    for tensor in outputs:
        assert isinstance(tensor, Buffer)
        assert np.allclose(expected, tensor.to(host).to_numpy(), atol=1e-6)


class FakeTransformerBlock(Module):
    """Mimics a distributed transformer block: two allreduce calls
    (attention + MLP) with per-device constants, designed to be invoked
    as a subgraph via ops.call.

    Pattern: scale -> allreduce -> residual -> scale -> allreduce -> residual
    """

    attn_allreduce: Allreduce
    mlp_allreduce: Allreduce
    num_devices: int

    def __init__(self, num_devices: int) -> None:
        super().__init__()
        self.attn_allreduce = Allreduce(num_accelerators=num_devices)
        self.mlp_allreduce = Allreduce(num_accelerators=num_devices)
        self.num_devices = num_devices

    def __call__(
        self,
        *args: TensorValue | BufferValue,
    ) -> list[TensorValue]:
        xs = [cast(TensorValue, arg) for arg in args[: self.num_devices]]
        signal_buffers = [
            cast(BufferValue, arg) for arg in args[self.num_devices :]
        ]

        # Per-device constants simulate RMSNorm weights (triggers hoisting)
        attn_scales = [
            ops.constant(2.0, dtype=DType.float32, device=DeviceRef.GPU(i))
            for i in range(self.num_devices)
        ]
        mlp_scales = [
            ops.constant(3.0, dtype=DType.float32, device=DeviceRef.GPU(i))
            for i in range(self.num_devices)
        ]

        # Attention path: scale -> allreduce -> residual add
        attn_out = [x * s for x, s in zip(xs, attn_scales, strict=True)]
        attn_out = self.attn_allreduce(attn_out, signal_buffers)
        hs = [x + a for x, a in zip(xs, attn_out, strict=True)]

        # MLP path: scale -> allreduce -> residual add
        mlp_out = [h * s for h, s in zip(hs, mlp_scales, strict=True)]
        mlp_out = self.mlp_allreduce(mlp_out, signal_buffers)
        hs = [h + m for h, m in zip(hs, mlp_out, strict=True)]

        return hs


@pytest.mark.parametrize("num_gpus", [2, 4, 8])
def test_transfer_to_subgraph_with_allreduce(num_gpus: int) -> None:
    """Tests transfers from GPU 0 -> subgraph with two allreduce calls.

    Mimics the Llama distributed transformer pattern:
    1. Input on GPU 0 is transferred to all peers (serialized rmo.mo.transfer)
    2. A subgraph (via ops.call) performs two allreduce operations with
       shared signal buffers and per-device constants
    3. Constants inside the subgraph trigger the HoistConstantSubgraphs pass

    This pattern exposed GEX-3097: the hoisting pass scrambled chain ordering
    at the subgraph boundary, causing GPU 0 to race ahead past transfers and
    deadlock in the allreduce barrier.
    """
    if (available_gpus := accelerator_count()) < num_gpus:
        pytest.skip(
            f"skipping {num_gpus=} test since only {available_gpus} available"
        )

    graph_devices = [DeviceRef.GPU(id) for id in range(num_gpus)]
    signals = Signals(devices=graph_devices)

    host = CPU()
    devices: list[Device] = [Accelerator(i) for i in range(num_gpus)]
    session = InferenceSession(devices=[host] + devices)

    model = FakeTransformerBlock(num_devices=num_gpus)

    with Graph(
        "TransferSubgraphAllreduce",
        input_types=[
            TensorType(DType.float32, shape=[M, N], device=DeviceRef.GPU(0)),
            *signals.input_types(),
        ],
    ) as graph:
        input_tensor = graph.inputs[0].tensor
        signal_bufs = [inp.buffer for inp in graph.inputs[1:]]

        # Serialized transfers from GPU 0 to all peers
        h = [input_tensor.to(DeviceRef.GPU(i)) for i in range(num_gpus)]

        # Build and invoke transformer block as a subgraph
        subgraph_input_types = [
            TensorType(DType.float32, shape=[M, N], device=graph_devices[i])
            for i in range(num_gpus)
        ] + list(signals.input_types())

        subgraph = model.build_subgraph(
            "transformer_block", subgraph_input_types
        )

        results = ops.call(subgraph, *h, *signal_bufs)
        graph.output(*[r.tensor for r in results])

    compiled = session.load(graph)

    input_np = np.ones((M, N), np.float32)
    input_buf = Buffer.from_numpy(input_np).to(devices[0])

    outputs = compiled.execute(input_buf, *signals.buffers())

    # Expected: after attn allreduce: 1 + 2*N, after MLP allreduce:
    # (1+2N) + 3N*(1+2N) = (1+2N)(1+3N)
    expected_val = (1.0 + 2.0 * num_gpus) * (1.0 + 3.0 * num_gpus)
    expected = np.full((M, N), expected_val, dtype=np.float32)

    for tensor in outputs:
        assert isinstance(tensor, Buffer)
        assert np.allclose(expected, tensor.to(host).to_numpy(), atol=1e-6)
