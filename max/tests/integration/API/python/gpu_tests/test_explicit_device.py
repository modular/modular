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
import pytest
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
        cuda_input = graph.inputs[2].to(Device.CUDA(0))  # type: ignore
        sum2 = ops.add(sum, cuda_input)
        graph.output(sum2)
    return graph


def create_test_graph_io_devices() -> Graph:
    cuda_input_type = TensorType(
        dtype=DType.float32, shape=["batch", "channels"], device=Device.CUDA(0)
    )
    cpu_input_type = TensorType(
        dtype=DType.float32, shape=["batch", "channels"], device=Device.CPU(0)
    )
    with Graph(
        "add",
        input_types=(
            cuda_input_type,
            cpu_input_type,
            cpu_input_type,
            cuda_input_type,
        ),
    ) as graph:
        cuda_input1 = graph.inputs[1].to(Device.CUDA(0))  # type: ignore
        cuda_input2 = graph.inputs[2].to(Device.CUDA(0))  # type: ignore
        sum = ops.add(graph.inputs[0], cuda_input1)
        sum2 = ops.add(sum, cuda_input2)
        graph.output(sum2)
    return graph


def test_io_device_properties() -> None:
    graph = create_test_graph_io_devices()
    host = CPU()
    cuda0 = CUDA(0)
    session = InferenceSession(devices=[host, cuda0])
    compiled = session.load(graph)
    assert len(compiled.output_devices) == 1
    assert str(cuda0) == str(compiled.output_devices[0])
    assert len(compiled.input_devices) == 4
    assert str(cuda0) == str(compiled.input_devices[0])
    assert str(host) == str(compiled.input_devices[1])
    assert str(host) == str(compiled.input_devices[2])
    assert str(cuda0) == str(compiled.input_devices[3])
    assert len(compiled.devices) == 2
    assert str(host) == str(compiled.devices[0])
    assert str(cuda0) == str(compiled.devices[1])


def test_io_device_output_errors() -> None:
    graph = create_test_graph_io_devices()
    host = CPU()
    cuda0 = CUDA(0)
    session = InferenceSession(devices=[host])
    with pytest.raises(
        ValueError,
        match=(
            "Loaded Model output idx=0 uses device=cuda:0 which was not set up"
            " in InferenceSession"
        ),
    ):
        compiled = session.load(graph)
        compiled.output_devices


def test_io_device_input_errors() -> None:
    graph = create_test_graph_io_devices()
    host = CPU()
    cuda0 = CUDA(0)
    session = InferenceSession(devices=[host])
    with pytest.raises(
        ValueError,
        match=(
            "Loaded Model input idx=0 uses device=cuda:0 which was not set up"
            " in InferenceSession"
        ),
    ):
        compiled = session.load(graph)
        compiled.input_devices


def test_explicit_device_compilation() -> None:
    graph = create_test_graph_with_transfer()
    device = CUDA(0)
    session = InferenceSession(devices=[device])
    compiled = session.load(graph)
    assert str(device) == str(compiled.devices[0])


def test_explicit_device_execution() -> None:
    graph = create_test_graph_with_transfer()
    host = CPU()
    device = CUDA(0)
    session = InferenceSession(devices=[device])
    compiled = session.load(graph)
    a_np = np.ones((1, 1)).astype(np.float32)
    b_np = np.ones((1, 1)).astype(np.float32)
    c_np = np.ones((1, 1)).astype(np.float32)
    a = Tensor.from_numpy(a_np).to(device)
    b = Tensor.from_numpy(b_np).to(device)
    c = Tensor.from_numpy(b_np)
    output = compiled.execute(a, b, c)
    assert np.allclose((a_np + b_np + c_np), output[0].to(host).to_numpy())  # type: ignore
