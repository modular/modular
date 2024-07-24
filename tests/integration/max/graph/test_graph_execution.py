# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.engine Python bindings with Max Graph."""


import max.engine as me
import numpy as np
from max.graph import DType, Graph, TensorType, ops


def test_max_graph():
    session = me.InferenceSession()
    input_type = TensorType(dtype=DType.float32, dims=["batch", "channels"])
    with Graph("add", input_types=(input_type, input_type)) as graph:
        graph.output(ops.add(graph.inputs[0], graph.inputs[1]))
        compiled = session.load(graph)
        a = np.ones((1, 1)).astype(np.float32)
        b = np.ones((1, 1)).astype(np.float32)
        output = compiled.execute(input0=a, input1=b)
        assert output["output0"] == a + b
