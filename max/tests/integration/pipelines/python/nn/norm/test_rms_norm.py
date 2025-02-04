# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import numpy as np
import pytest
import torch
from max.dtype import DType
from max.graph import Graph, TensorType
from max.pipelines.nn import RMSNorm
from modular_graph_test import are_all_tensor_values, modular_graph_test


def torch_rms_norm(x, weight, eps=1e-6):
    #   See https://github.com/meta-llama/llama/blob/main/llama/model.py#L34
    return x * torch.rsqrt((x**2).mean(-1, keepdim=True) + eps) * weight


@pytest.mark.parametrize(
    "input_type",
    [
        TensorType(DType.float32, ["dim"]),
        TensorType(DType.float32, ["batch", "dim"]),
        TensorType(DType.float32, ["a", "x", "y", "z", "dim"]),
        TensorType(DType.float64, ["dim"]),
    ],
)
def test_norm(session, input_type):
    # Initialize Graph
    dim = input_type.shape[-1]
    weight_type = TensorType(input_type.dtype, [dim])
    with Graph("norm", input_types=[input_type, weight_type]) as graph:
        assert are_all_tensor_values(graph.inputs)
        x, weight = graph.inputs
        graph.output(RMSNorm(weight)(x))

        @modular_graph_test(session, graph)
        def test_correctness(execute, inputs, torch_inputs):
            result = execute(inputs)
            expected = torch_rms_norm(*torch_inputs).detach().numpy()
            ACCURACY_RTOL = 1e-2
            ACCURACY_ATOL = 1e-8
            np.testing.assert_allclose(
                result,
                expected,
                atol=ACCURACY_ATOL,
                rtol=ACCURACY_RTOL,
                equal_nan=True,
            )
