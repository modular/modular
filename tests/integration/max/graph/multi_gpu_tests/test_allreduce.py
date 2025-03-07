# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Test the max.engine Python bindings with Max Graph when using explicit device."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Device, Tensor
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
from max.pipelines.nn import Allreduce, LayerV2, Signals

M = 512
N = 1024


def allreduce_graph(signals: Signals) -> Graph:
    devices = signals.devices
    with Graph(
        "allreduce",
        input_types=[
            TensorType(dtype=DType.float32, shape=[M, N], device=devices[0]),
            TensorType(dtype=DType.float32, shape=[M, N], device=devices[1]),
            TensorType(dtype=DType.float32, shape=[M, N], device=devices[2]),
            TensorType(dtype=DType.float32, shape=[M, N], device=devices[3]),
            *signals.input_types(),
        ],
    ) as graph:
        assert isinstance(graph.inputs[0], TensorValue)
        assert isinstance(graph.inputs[1], TensorValue)
        assert isinstance(graph.inputs[2], TensorValue)
        assert isinstance(graph.inputs[3], TensorValue)
        add0 = graph.inputs[0]
        add1 = graph.inputs[1] * 2
        add2 = graph.inputs[2] * 3
        add3 = graph.inputs[3] * 4

        allreduce = Allreduce(num_accelerators=len(devices))
        allreduce_outputs = allreduce(
            [add0, add1, add2, add3],
            [inp.buffer for inp in graph.inputs[4:]],
        )

        graph.output(
            allreduce_outputs[0],
            allreduce_outputs[1],
            allreduce_outputs[2],
            allreduce_outputs[3],
        )
        return graph


def test_allreduce_execution() -> None:
    """Tests multi-device allreduce execution."""
    num_gpus = 4
    signals = Signals(devices=[DeviceRef.GPU(id=id) for id in range(num_gpus)])
    graph = allreduce_graph(signals)
    host = CPU()
    device0 = Accelerator(0)
    device1 = Accelerator(1)
    device2 = Accelerator(2)
    device3 = Accelerator(3)
    session = InferenceSession(
        devices=[host, device0, device1, device2, device3]
    )
    compiled = session.load(graph)
    a_np = np.ones((M, N)).astype(np.float32)
    out_np = a_np * 10
    a = Tensor.from_numpy(a_np).to(device0)
    b = Tensor.from_numpy(a_np).to(device1)
    c = Tensor.from_numpy(a_np).to(device2)
    d = Tensor.from_numpy(a_np).to(device3)

    # Synchronize devices so that the signal buffers are initialized.
    for dev in (device0, device1, device2, device3):
        dev.synchronize()

    output = compiled.execute(a, b, c, d, *signals.buffers())

    # Check Executed Graph
    assert isinstance(output[0], Tensor)
    assert output[0].device == device0
    assert np.allclose(out_np, output[0].to(host).to_numpy())
    assert isinstance(output[1], Tensor)
    assert output[1].device == device1
    assert np.allclose(out_np, output[1].to(host).to_numpy())
    assert isinstance(output[2], Tensor)
    assert output[2].device == device2
    assert np.allclose(out_np, output[2].to(host).to_numpy())
    assert isinstance(output[3], Tensor)
    assert output[3].device == device3
    assert np.allclose(out_np, output[3].to(host).to_numpy())


class AllreduceAddBase(LayerV2):
    """Base class for allreduce variants with elementwise add."""

    num_devices: int
    """Number of devices to allreduce between."""

    def __init__(self, num_devices: int) -> None:
        super().__init__()
        self.num_devices = num_devices

    def __call__(
        self,
        *args: TensorValue | BufferValue,
    ) -> list[TensorValue]:
        raise NotImplementedError


class AllreduceAddNaive(AllreduceAddBase):
    """A naive allreduce with an elementwise add."""

    def __call__(
        self,
        *args: TensorValue | BufferValue,
    ) -> list[TensorValue]:
        inputs = [cast(TensorValue, arg) for arg in args[: self.num_devices]]

        # Allreduce implementation using device-to-device transfers.
        results = ops.allreduce.sum_naive(inputs)
        return [x + 42 for x in results]


class AllreduceAdd(AllreduceAddBase):
    """A fused allreduce with an elementwise add."""

    def __init__(self, num_devices: int) -> None:
        super().__init__(num_devices)
        self.allreduce = Allreduce(num_accelerators=num_devices)

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
            ops.constant(42, dtype=DType.float32).to(DeviceRef.GPU(id))
            for id in range(self.num_devices)
        ]

        # Elementwise add that should fuse into allreduce's epilogue.
        return [x + y for x, y in zip(results, biases)]


@pytest.mark.parametrize("layer_cls", [AllreduceAdd, AllreduceAddNaive])
@pytest.mark.parametrize("num_gpus", [1, 2, 4])
def test_allreduce_epilogue_fusion(
    layer_cls: type[AllreduceAddBase], num_gpus: int
) -> None:
    """Tests that an elementwise add correctly follows an allreduce operation."""
    graph_devices = [DeviceRef.GPU(id) for id in range(num_gpus)]
    signals = Signals(devices=graph_devices)

    host = CPU()
    devices: list[Device] = [Accelerator(i) for i in range(num_gpus)]
    session = InferenceSession(devices=[host] + devices)

    model = layer_cls(num_devices=len(devices))
    graph = Graph(
        f"{layer_cls.__name__}_fusion",
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
