# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
import tempfile

import numpy as np
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, ops


def test_graph_telemetry():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        filepath = temp_file.name
    os.environ["MODULAR_TELEMETRY_EXPORTERS_LOGS_FILE_PATH"] = filepath
    input_type = TensorType(dtype=DType.float32, shape=["batch", "channels"])
    session = InferenceSession()
    with Graph("add", input_types=(input_type, input_type)) as graph:
        graph.output(ops.add(graph.inputs[0], graph.inputs[1]))
        compiled = session.load(graph)
        a = np.ones((1, 1)).astype(np.float32)
        b = np.ones((1, 1)).astype(np.float32)
        _ = compiled.execute(a, b)

    expected_line = "max.pipeline.name: add"
    with open(filepath, "r") as file:
        lines = [line.strip() for line in file.readlines()]
    assert expected_line in lines
