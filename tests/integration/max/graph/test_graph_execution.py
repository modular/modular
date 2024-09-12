# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.engine Python bindings with Max Graph."""

import os
import tempfile

import numpy as np
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, ops


def test_max_graph(session):
    input_type = TensorType(dtype=DType.float32, shape=["batch", "channels"])
    with Graph("add", input_types=(input_type, input_type)) as graph:
        graph.output(ops.add(graph.inputs[0], graph.inputs[1]))
        compiled = session.load(graph)
        a_np = np.ones((1, 1), dtype=np.float32)
        a = Tensor.from_numpy(a_np)
        b_np = np.ones((1, 1), dtype=np.float32)
        b = Tensor.from_numpy(b_np)
        output = compiled.execute(a, b)
        assert np.allclose((a_np + b_np), output[0].to_numpy())


def test_max_graph_export(session):
    """Creates a graph via max-graph API, exports the mef to a tempfile, and
    check to ensure that the file contents are non-empty."""
    input_type = TensorType(dtype=DType.float32, shape=["batch", "channels"])
    with tempfile.NamedTemporaryFile() as mef_file:
        with Graph("add", input_types=(input_type, input_type)) as graph:
            graph.output(ops.add(graph.inputs[0], graph.inputs[1]))
            compiled = session.load(graph)
            compiled._export_mef(mef_file.name)
            assert os.path.getsize(mef_file.name) > 0


def test_max_graph_export_import(session):
    """Creates a graph via max-graph API, exports the mef to a tempfile, and
    loads the mef. Both the original model from the max-graph and the model
    from the mef are executed to ensure that they produce the same output."""
    input_type = TensorType(dtype=DType.float32, shape=["batch", "channels"])
    with tempfile.NamedTemporaryFile() as mef_file:
        with Graph("add", input_types=(input_type, input_type)) as graph:
            graph.output(ops.add(graph.inputs[0], graph.inputs[1]))
            compiled = session.load(graph)
            compiled._export_mef(mef_file.name)
            a_np = np.ones((1, 1)).astype(np.float32)
            b_np = np.ones((1, 1)).astype(np.float32)
            a = Tensor.from_numpy(a_np)
            b = Tensor.from_numpy(b_np)
            output = compiled.execute(a, b)
            assert np.allclose((a_np + b_np), output[0].to_numpy())
            compiled2 = session.load(mef_file.name)
            # Executing a mef-loaded model with a device tensor seems to not work.
            output2 = compiled2.execute(a_np, b_np)
            assert np.allclose((a_np + b_np), output2[0].to_numpy())
            assert np.allclose(output[0].to_numpy(), output2[0].to_numpy())


def test_max_graph_device():
    input_type = TensorType(dtype=DType.float32, shape=["batch", "channels"])
    with Graph("add", input_types=(input_type, input_type)) as graph:
        graph.output(ops.add(graph.inputs[0], graph.inputs[1]))
        device = CPU()
        session = InferenceSession(device=device)
        compiled = session.load(graph)
        assert str(device) == str(compiled.device)


def test_identity(session):
    # Create identity graph.
    graph = Graph(
        "identity",
        lambda x: x,
        input_types=[TensorType(DType.int32, (1,))],
    )

    # Compile and execute identity.
    model = session.load(graph)
    input = Tensor(shape=(1,), dtype=DType.int32)
    output = model.execute(input)

    # Test that using output's storage is still valid after destroying input.
    del input
    _ = output[0][0]
