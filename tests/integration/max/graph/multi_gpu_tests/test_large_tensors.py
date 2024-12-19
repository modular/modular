# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.engine supporting when graphs don't fit on a single-GPU."""

import numpy as np
from max.driver import CPU, Accelerator, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops


def create_multi_device_graph() -> Graph:
    m0_type = TensorType(
        dtype=DType.float32, shape=["m", "k"], device=DeviceRef.GPU(0)
    )
    n0_type = TensorType(
        dtype=DType.float32, shape=["k", "n"], device=DeviceRef.GPU(0)
    )
    m1_type = TensorType(
        dtype=DType.float32, shape=["m", "k"], device=DeviceRef.GPU(1)
    )
    n1_type = TensorType(
        dtype=DType.float32, shape=["k", "n"], device=DeviceRef.GPU(1)
    )
    with Graph(
        "DistributedMatmul", input_types=(m0_type, n0_type, m1_type, n1_type)
    ) as graph:
        m0, n0, m1, n1 = graph.inputs
        assert isinstance(m0, TensorValue)
        assert isinstance(n0, TensorValue)
        assert isinstance(m1, TensorValue)
        assert isinstance(n1, TensorValue)
        mm1 = m0 @ n0
        mm2 = m1 @ n1
        graph.output(*ops.allreduce.sum([mm1, mm2]))
    return graph


def test_gpu_io_graph_execution() -> None:
    """Tests multi-device transfers where inputs/outputs are on cpu."""
    graph = create_multi_device_graph()
    # Check built graph
    assert str(DeviceRef.GPU(0)) in str(graph)
    assert str(DeviceRef.GPU(1)) in str(graph)
    host = CPU()
    device0 = Accelerator(0)
    device1 = Accelerator(1)
    session = InferenceSession(devices=[device0, device1])
    compiled = session.load(graph)
    m_np = np.ones((1000, 1000000)).astype(np.float32)  # Large AF
    n_np = np.ones((1000000, 1000)).astype(np.float32)
    m0 = Tensor.from_numpy(m_np).to(device0)
    n0 = Tensor.from_numpy(n_np).to(device0)
    m1 = Tensor.from_numpy(m_np).to(device1)
    n1 = Tensor.from_numpy(n_np).to(device1)
    output = compiled.execute(m0, n0, m1, n1)
    assert isinstance(output[0], Tensor)
    assert isinstance(output[1], Tensor)
    # Check Executed Graph
    exp_output = np.matmul(m_np, n_np) * 2
    assert np.allclose(exp_output, output[0].to(host).to_numpy())
    assert np.allclose(exp_output, output[1].to(host).to_numpy())
