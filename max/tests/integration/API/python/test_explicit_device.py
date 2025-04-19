# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.engine Python bindings with Max Graph when using explicit device."""

import pytest
from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops


@pytest.mark.skipif(
    accelerator_count() == 0, reason="Requires gpu device to test"
)
def create_test_graph_with_transfer() -> Graph:
    input_type = TensorType(
        dtype=DType.float32,
        shape=["batch", "channels"],
        device=DeviceRef.GPU(0),
    )
    with Graph(
        "add", input_types=(input_type, input_type, input_type)
    ) as graph:
        sum = ops.add(graph.inputs[0], graph.inputs[1])
        assert isinstance(graph.inputs[2], TensorValue)
        cuda_input = graph.inputs[2].to(DeviceRef.GPU(0))
        sum2 = ops.add(sum, cuda_input)
        graph.output(sum2)
    return graph


@pytest.mark.skipif(
    accelerator_count() == 0, reason="Requires gpu device to test"
)
def create_test_graph_io_devices() -> Graph:
    cuda_input_type = TensorType(
        dtype=DType.float32,
        shape=["batch", "channels"],
        device=DeviceRef.GPU(0),
    )
    cpu_input_type = TensorType(
        dtype=DType.float32,
        shape=["batch", "channels"],
        device=DeviceRef.CPU(0),
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
        assert isinstance(graph.inputs[1], TensorValue)
        assert isinstance(graph.inputs[2], TensorValue)
        cuda_input1 = graph.inputs[1].to(DeviceRef.GPU(0))
        cuda_input2 = graph.inputs[2].to(DeviceRef.GPU(0))
        sum = ops.add(graph.inputs[0], cuda_input1)
        sum2 = ops.add(sum, cuda_input2)
        graph.output(sum2)
    return graph


@pytest.mark.skipif(
    accelerator_count() == 0, reason="Requires gpu device to test"
)
def test_io_device_properties() -> None:
    graph = create_test_graph_io_devices()
    host = CPU()
    cuda0 = Accelerator(0)
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


@pytest.mark.skipif(
    accelerator_count() == 0, reason="Requires gpu device to test"
)
def test_io_device_output_errors() -> None:
    graph = create_test_graph_io_devices()
    host = CPU()
    cuda0 = Accelerator(0)
    session = InferenceSession(devices=[host])
    with pytest.raises(
        ValueError,
        match=(
            "Loaded Model .* uses device=gpu:0 which was not set up"
            " in InferenceSession"
        ),
    ):
        compiled = session.load(graph)


@pytest.mark.skipif(
    accelerator_count() == 0, reason="Requires gpu device to test"
)
def test_explicit_device_compilation() -> None:
    graph = create_test_graph_with_transfer()
    device = Accelerator(0)
    session = InferenceSession(devices=[device])
    compiled = session.load(graph)
    assert str(device) == str(compiled.devices[0])
