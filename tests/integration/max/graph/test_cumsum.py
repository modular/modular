# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

#
# ===----------------------------------------------------------------------=== #

import numpy as np
import torch
from max.dtype import DType
from max.graph import Graph, TensorType, ops
from modular_graph_test import modular_graph_test


def test_cumsum(session):
    input_type = TensorType(DType.float32, [1024])

    with Graph("cumsum", input_types=[input_type]) as graph:
        out = ops.cumsum(graph.inputs[0], axis=0)
        graph.output(out)

    @modular_graph_test(session, graph)
    def test_correctness(execute, inputs, torch_inputs):
        max_result = execute(inputs)
        torch_result = torch.cumsum(torch_inputs[0], dim=0).numpy()
        np.testing.assert_allclose(
            max_result, torch_result, rtol=1e-6, atol=1e-6
        )
