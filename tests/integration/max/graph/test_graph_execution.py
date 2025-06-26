# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.engine Python bindings with Max Graph."""

import os
import tempfile
from pathlib import Path

import numpy as np
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def create_test_graph():
    input_type = TensorType(
        dtype=DType.float32, shape=["batch", "channels"], device=DeviceRef.CPU()
    )
    with Graph("add", input_types=(input_type, input_type)) as graph:
        graph.output(ops.add(graph.inputs[0], graph.inputs[1]))
    return graph


def test_max_graph(session):
    graph = create_test_graph()
    compiled = session.load(graph)
    a_np = np.ones((1, 1)).astype(np.float32)
    b_np = np.ones((1, 1)).astype(np.float32)
    a = Tensor.from_numpy(a_np).to(compiled.input_devices[0])
    b = Tensor.from_numpy(b_np).to(compiled.input_devices[1])
    output = compiled.execute(a, b)
    assert np.allclose((a_np + b_np), output[0].to_numpy())


def test_max_graph_export(session):
    """Creates a graph via max-graph API, exports the mef to a tempfile, and
    check to ensure that the file contents are non-empty."""

    with tempfile.NamedTemporaryFile() as mef_file:
        graph = create_test_graph()
        compiled = session.load(graph)
        compiled._export_mef(mef_file.name)
        assert os.path.getsize(mef_file.name) > 0


def test_max_graph_export_import_mef(session):
    """Creates a graph via max-graph API, exports the mef to a tempfile, and
    loads the mef. Both the original model from the max-graph and the model
    from the mef are executed to ensure that they produce the same output."""

    with tempfile.NamedTemporaryFile() as mef_file:
        graph = create_test_graph()
        compiled = session.load(graph)
        compiled._export_mef(mef_file.name)
        a_np = np.ones((1, 1)).astype(np.float32)
        b_np = np.ones((1, 1)).astype(np.float32)
        a = Tensor.from_numpy(a_np).to(compiled.input_devices[0])
        b = Tensor.from_numpy(b_np).to(compiled.input_devices[1])
        output = compiled.execute(a, b)[0].to_numpy()
        assert np.allclose((a_np + b_np), output)
        compiled2 = session.load(mef_file.name)
        # Executing a mef-loaded model with a device tensor seems to not work.
        output2 = compiled2.execute(a, b)[0].to_numpy()
        assert np.allclose((a_np + b_np), output2)
        assert np.allclose(output, output2)


def test_max_graph_device(session):
    graph = create_test_graph()
    device = CPU()
    session = InferenceSession(devices=[device])
    compiled = session.load(graph)
    assert str(device) == str(compiled.devices[0])


def test_identity(session):
    # Create identity graph.
    graph = Graph(
        "identity",
        lambda x: x,
        input_types=[TensorType(DType.int32, (1,), device=DeviceRef.CPU())],
    )

    # Compile and execute identity.
    model = session.load(graph)
    input = Tensor(shape=(1,), dtype=DType.int32)
    output = model.execute(input.to(model.input_devices[0]))
    # Test that using output's storage is still valid after destroying input.
    del input
    _ = output[0].to(CPU())[0]


def test_max_graph_export_import_mlir(session):
    """Creates a graph via max-graph API, exports the mlir to a tempfile, and
    loads the mlir. Both the original model from the max-graph and the model
    from the mlir are executed to ensure that they produce the same output."""

    with tempfile.NamedTemporaryFile(mode="w+") as mlir_file:
        graph = create_test_graph()
        compiled = session.load(graph)
        mlir_file.write(str(graph._module))
        a_np = np.ones((1, 1)).astype(np.float32)
        b_np = np.ones((1, 1)).astype(np.float32)
        a = Tensor.from_numpy(a_np).to(compiled.input_devices[0])
        b = Tensor.from_numpy(b_np).to(compiled.input_devices[1])
        output = compiled.execute(a, b)[0].to_numpy()
        assert output == a_np + b_np

        mlir_file.flush()

        # Now load the model from mlir.
        graph2 = Graph(name="add", path=Path(mlir_file.name))
        compiled2 = session.load(graph2)
        output2 = compiled2.execute(a, b)[0].to_numpy()
        assert output2 == a_np + b_np
        assert output == output2
