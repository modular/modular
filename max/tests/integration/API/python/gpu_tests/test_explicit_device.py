# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.engine Python bindings with Max Graph when using explicit device."""

import os
import tempfile
from pathlib import Path

import numpy as np
from max.driver import CPU, CUDA, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Device, Graph, TensorType, ops


def create_test_graph_with_transfer() -> Graph:
    input_type = TensorType(
        dtype=DType.float32, shape=["batch", "channels"], device=Device.CUDA(0)
    )
    with Graph(
        "add", input_types=(input_type, input_type, input_type)
    ) as graph:
        sum = ops.add(graph.inputs[0], graph.inputs[1])
        cuda_input = graph.inputs[2].to(Device.CUDA(0))
        sum2 = ops.add(sum, cuda_input)
        graph.output(sum2)
    return graph


def test_explicit_device_compilation() -> None:
    graph = create_test_graph_with_transfer()
    device = CUDA(0)
    session = InferenceSession(device=device)
    compiled = session.load(graph)
    assert str(device) == str(compiled.device)


def test_explicit_device_execution() -> None:
    graph = create_test_graph_with_transfer()
    host = CPU()
    device = CUDA(0)
    session = InferenceSession(device=device)
    compiled = session.load(graph)
    a_np = np.ones((1, 1)).astype(np.float32)
    b_np = np.ones((1, 1)).astype(np.float32)
    c_np = np.ones((1, 1)).astype(np.float32)
    a = Tensor.from_numpy(a_np).to(device)
    b = Tensor.from_numpy(b_np).to(device)
    c = Tensor.from_numpy(b_np)
    output = compiled.execute(a, b, c)
    assert np.allclose((a_np + b_np + c_np), output[0].to(host).to_numpy())
