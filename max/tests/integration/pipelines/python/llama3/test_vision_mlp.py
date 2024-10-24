# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs Llama3.2 vision MLP and compares it against the transformers package
reference implementation.
"""

from dataclasses import dataclass
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from max.dtype import DType
from max.graph import Graph, TensorType
from modular_graph_test import modular_graph_test
from nn import Linear
from llama3.vision.mlp import MLP


def torch_linear(weight, **kwargs):
    linear = nn.Linear(*weight.shape, **kwargs)
    linear.weight = nn.Parameter(weight)
    return linear


class TorchMLP(nn.Module):
    def __init__(self, w1, w2):
        super().__init__()
        self.fc1 = torch_linear(w1, bias=False)
        self.fc2 = torch_linear(w2, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


@pytest.mark.parametrize(
    "input_type",
    [
        TensorType(DType.float32, ["dim"]),
        TensorType(DType.float32, ["batch", "dim"]),
        TensorType(DType.float32, ["x", "y", "z", "dim"]),
        # TODO(GRA-855): batched matmul rank > 4
        # TensorType(DType.float32, ["a", "x", "y", "z", "dim"]),
        TensorType(DType.float64, ["dim"]),
    ],
)
def test_mlp(session, input_type: TensorType):
    dim = input_type.shape[-1]
    w1_type = TensorType(input_type.dtype, ["hidden_dim", dim])
    w2_type = TensorType(input_type.dtype, [dim, "hidden_dim"])
    with Graph("mlp", input_types=[input_type, w1_type, w2_type]) as graph:
        x, w1, w2 = graph.inputs
        mlp = MLP(Linear(w1), Linear(w2))
        graph.output(mlp(x))

        # This is set so it fits a float type with width of 32.
        @modular_graph_test(session, graph, max_magnitude=1 / 64)
        def test_correctness(execute, inputs, torch_inputs):
            result = execute(inputs)
            x, w1, w2 = torch_inputs

            # Transpose weights to match our Linear semantics.
            expected = TorchMLP(w1, w2)(x).detach().numpy()
            # TODO(MSDK-1071): Consolidate and figure out how to call
            # assert_allclose(result, expected) to fire again on mismatched
            # tensor values.
            ACCURACY_RTOL = 1e-2
            ACCURACY_ATOL = 1e-8
            np.testing.assert_allclose(
                result,
                expected,
                atol=ACCURACY_ATOL,
                rtol=ACCURACY_RTOL,
                equal_nan=True,
            )
