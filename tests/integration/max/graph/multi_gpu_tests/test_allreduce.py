# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Test the max.engine Python bindings with Max Graph when using explicit device."""

from __future__ import annotations

import numpy as np
from max.driver import CPU, Accelerator, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.pipelines.nn import Signals


def allreduce_graph(signals: Signals) -> Graph:
    devices = signals.devices
    with Graph(
        "allreduce",
        input_types=[
            TensorType(
                dtype=DType.float32, shape=[30, 1000], device=devices[0]
            ),
            TensorType(
                dtype=DType.float32, shape=[30, 1000], device=devices[1]
            ),
            TensorType(
                dtype=DType.float32, shape=[30, 1000], device=devices[2]
            ),
            TensorType(
                dtype=DType.float32, shape=[30, 1000], device=devices[3]
            ),
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
        allreduce_outputs = ops.allreduce.sum(
            inputs=[add0, add1, add2, add3],
            signal_buffers=[inp.buffer for inp in graph.inputs[4:]],
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
    signals = Signals(
        devices=[
            DeviceRef.GPU(id=0),
            DeviceRef.GPU(id=1),
            DeviceRef.GPU(id=2),
            DeviceRef.GPU(id=3),
        ]
    )
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
    a_np = np.ones((30, 1000)).astype(np.float32)
    out_np = a_np * 10
    a = Tensor.from_numpy(a_np).to(device0)
    b = Tensor.from_numpy(a_np).to(device1)
    c = Tensor.from_numpy(a_np).to(device2)
    d = Tensor.from_numpy(a_np).to(device3)

    signal_buffers = [
        Tensor.zeros(
            shape=(Signals.NUM_BYTES,),
            dtype=DType.uint8,
            device=dev,
        )
        for dev in (device0, device1, device2, device3)
    ]

    # Synchronize devices so that the signal buffers are initialized.
    for dev in (device0, device1, device2, device3):
        dev.synchronize()

    output = compiled.execute(a, b, c, d, *signal_buffers)

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
