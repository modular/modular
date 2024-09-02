# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import pytest
import torch.nn as nn
import torch.nn.functional as F
from conftest import assert_allclose, modular_graph_test
from llama3.model.mlp import MLP, Linear
from max.dtype import DType
from max.graph import Graph, TensorType


def torch_linear(weight, **kwargs):
    linear = nn.Linear(*weight.shape, **kwargs)
    linear.weight = nn.Parameter(weight)
    return linear


class TorchMLP(nn.Module):
    def __init__(self, w1, w2, w3):
        super().__init__()
        self.gate_proj = torch_linear(w1, bias=False)
        self.down_proj = torch_linear(w2, bias=False)
        self.up_proj = torch_linear(w3, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


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
    w3_type = w1_type
    with Graph(
        "mlp", input_types=[input_type, w1_type, w2_type, w3_type]
    ) as graph:
        x, w1, w2, w3 = graph.inputs
        mlp = MLP(Linear(w1), Linear(w2), Linear(w3))
        graph.output(mlp(x))

        @modular_graph_test(session, graph)
        def test_correctness(execute, inputs, torch_inputs):
            result = execute(inputs)
            x, w1, w2, w3 = torch_inputs
            # Transpose weights to match our Linear semantics.
            expected = TorchMLP(w1, w2, w3)(x).detach().numpy()
            assert_allclose(result, expected)
