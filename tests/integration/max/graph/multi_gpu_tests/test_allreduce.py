# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# type: ignore

"""Test the max.engine Python bindings with Max Graph when using explicit device."""


import numpy as np
from max.driver import CPU, CUDA, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Device, Graph, TensorType, ops


def allreduce_graph() -> Graph:
    devices = [
        Device.CUDA(id=0),
        Device.CUDA(id=1),
        Device.CUDA(id=2),
        Device.CUDA(id=3),
    ]
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
        ],
    ) as graph:
        add0 = graph.inputs[0]
        add1 = graph.inputs[1] * 2
        add2 = graph.inputs[2] * 3
        add3 = graph.inputs[3] * 4
        allreduce_outputs = ops.allreduce.sum([add0, add1, add2, add3])
        graph.output(
            allreduce_outputs[0],
            allreduce_outputs[1],
            allreduce_outputs[2],
            allreduce_outputs[3],
        )
        return graph


def test_allreduce_execution() -> None:
    """Tests multi-device allreduce execution."""
    graph = allreduce_graph()
    host = CPU()
    device0 = CUDA(0)
    device1 = CUDA(1)
    device2 = CUDA(2)
    device3 = CUDA(3)
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
    output = compiled.execute(a, b, c, d)
    # Check Executed Graph
    assert output[0].device == device0
    assert np.allclose(out_np, output[0].to(host).to_numpy())
    assert output[1].device == device1
    assert np.allclose(out_np, output[1].to(host).to_numpy())
    assert output[2].device == device2
    assert np.allclose(out_np, output[2].to(host).to_numpy())
    assert output[3].device == device3
    assert np.allclose(out_np, output[3].to(host).to_numpy())
