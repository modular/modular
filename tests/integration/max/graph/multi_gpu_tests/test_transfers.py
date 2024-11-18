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


def create_multi_device_graph_with_cpu_io() -> Graph:
    input_type = TensorType(
        dtype=DType.float32, shape=["batch", "channels"], device=Device.CPU()
    )
    with Graph(
        "add", input_types=(input_type, input_type, input_type)
    ) as graph:
        gpu0_input0 = graph.inputs[0].to(Device.CUDA(0))  # type: ignore
        gpu1_input1 = graph.inputs[1].to(Device.CUDA(1))  # type: ignore
        gpu0_input2 = graph.inputs[2].to(Device.CUDA(0))  # type: ignore
        gpu0_input1 = gpu1_input1.to(Device.CUDA(0))
        sum = ops.add(gpu0_input0, gpu0_input2)
        sum2 = ops.add(sum, gpu0_input1)
        cpu_output = sum2.to(Device.CPU())
        graph.output(cpu_output)
    return graph


def create_multi_device_graph_with_gpu_io() -> Graph:
    input_type0 = TensorType(
        dtype=DType.float32,
        shape=["batch", "channels"],
        device=Device.CUDA(id=0),
    )
    input_type1 = TensorType(
        dtype=DType.float32,
        shape=["batch", "channels"],
        device=Device.CUDA(id=1),
    )
    with Graph(
        "add", input_types=(input_type0, input_type1, input_type0)
    ) as graph:
        gpu0_input1 = graph.inputs[1].to(Device.CUDA(0))  # type: ignore
        sum = ops.add(graph.inputs[0], graph.inputs[2])
        sum2 = ops.add(sum, gpu0_input1)
        graph.output(sum2)
    return graph


def test_cpu_io_graph_execution() -> None:
    """Tests multi-device transfers where inputs/outputs are on cpu."""
    graph = create_multi_device_graph_with_cpu_io()
    # Check built graph
    assert str(Device.CPU(0)) in str(graph)
    assert str(Device.CUDA(0)) in str(graph)
    assert str(Device.CUDA(1)) in str(graph)
    host = CPU()
    device0 = CUDA(0)
    device1 = CUDA(1)
    session = InferenceSession(devices=[host, device0, device1])
    compiled = session.load(graph)
    # Check Compiled Graph
    assert str(host) == str(compiled.devices[0])
    assert str(device0) == str(compiled.devices[1])
    assert str(device1) == str(compiled.devices[2])
    a_np = np.ones((1, 1)).astype(np.float32)
    b_np = np.ones((1, 1)).astype(np.float32)
    c_np = np.ones((1, 1)).astype(np.float32)
    a = Tensor.from_numpy(a_np)
    b = Tensor.from_numpy(b_np)
    c = Tensor.from_numpy(b_np)
    output = compiled.execute(a, b, c)
    # Check Executed Graph
    assert np.allclose((a_np + b_np + c_np), output[0].to_numpy())  # type: ignore


def test_gpu_io_graph_execution() -> None:
    """Tests multi-device transfers where inputs/outputs are on cpu."""
    graph = create_multi_device_graph_with_gpu_io()
    # Check built graph
    assert str(Device.CUDA(0)) in str(graph)
    assert str(Device.CUDA(1)) in str(graph)
    host = CPU()
    device0 = CUDA(0)
    device1 = CUDA(1)
    session = InferenceSession(devices=[host, device0, device1])
    compiled = session.load(graph)
    # Check Compiled Graph
    assert str(host) == str(compiled.devices[0])
    assert str(device0) == str(compiled.devices[1])
    assert str(device1) == str(compiled.devices[2])
    a_np = np.ones((1, 1)).astype(np.float32)
    b_np = np.ones((1, 1)).astype(np.float32)
    c_np = np.ones((1, 1)).astype(np.float32)
    a = Tensor.from_numpy(a_np).to(device0)
    b = Tensor.from_numpy(b_np).to(device1)
    c = Tensor.from_numpy(b_np).to(device0)
    output = compiled.execute(a, b, c)
    # Check Executed Graph
    assert np.allclose((a_np + b_np + c_np), output[0].to(host).to_numpy())  # type: ignore
