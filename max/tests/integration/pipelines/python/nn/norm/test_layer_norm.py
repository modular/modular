# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
from max.dtype import DType
from max.graph import Graph, TensorType
from modular_graph_test import assert_allclose, modular_graph_test
from nn import LPLayerNorm


def torch_layer_norm(x, weight):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)

    # Apply layer normalization.
    normalized_tensor = (x - mean) / (
        std + 1e-6
    )  # Adding epsilon for numerical stability.

    after_weights = normalized_tensor * weight

    return after_weights


@pytest.mark.parametrize(
    "input_type",
    [
        TensorType(DType.float32, ["dim"]),
        TensorType(DType.float32, ["batch", "dim"]),
        TensorType(DType.float32, ["x", "y", "z", "dim"]),
    ],
)
def test_layer_norm(session, input_type):
    # Initialize Graph
    dim = input_type.shape[-1]
    weight_type = TensorType(input_type.dtype, [dim])
    with Graph("layer_norm", input_types=[input_type, weight_type]) as graph:
        x, weight = graph.inputs
        graph.output(LPLayerNorm(weight)(x))

        @modular_graph_test(session, graph)
        def test_correctness(execute, inputs, torch_inputs):
            result = execute(inputs)
            expected = torch_layer_norm(*torch_inputs).detach().numpy()
            assert_allclose(result, expected)
