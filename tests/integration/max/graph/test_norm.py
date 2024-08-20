# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from conftest import modular_vs_torch_test
import pytest
import torch
from llama3.norm import RMSNorm
from max.graph import DType, Graph, TensorType


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
        x, weight = graph.inputs
        graph.output(RMSNorm(weight)(x))

        modular_vs_torch_test(session, graph, torch_rms_norm)
