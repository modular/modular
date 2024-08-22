# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.engine Python bindings with Max Graph."""


import os
import tempfile

import numpy as np
from max.dtype import DType
from max.graph import Graph, TensorType, ops


def test_max_graph(session):
    input_type = TensorType(dtype=DType.float32, shape=["batch", "channels"])
    with Graph("add", input_types=(input_type, input_type)) as graph:
        graph.output(ops.add(graph.inputs[0], graph.inputs[1]))
        compiled = session.load(graph)
        a = np.ones((1, 1)).astype(np.float32)
        b = np.ones((1, 1)).astype(np.float32)
        output = compiled.execute(input0=a, input1=b)
        assert output["output0"] == a + b


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
            a = np.ones((1, 1)).astype(np.float32)
            b = np.ones((1, 1)).astype(np.float32)
            output = compiled.execute(input0=a, input1=b)
            assert output["output0"] == a + b
            compiled2 = session.load(mef_file.name)
            output2 = compiled2.execute(input0=a, input1=b)
            assert output2["output0"] == a + b
            assert output["output0"] == output2["output0"]
