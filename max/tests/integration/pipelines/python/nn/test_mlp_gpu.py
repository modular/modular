# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from typing import List

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from max.driver import CPU, Accelerator, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.nn import MLP, Linear
from modular_graph_test import are_all_tensor_values


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


def mlp_graph(types: List[TensorType]) -> Graph:
    (input_type, w1_type, w2_type, w3_type) = types
    with Graph(
        "mlp", input_types=[input_type, w1_type, w2_type, w3_type]
    ) as graph:
        assert are_all_tensor_values(graph.inputs)
        x, w1, w2, w3 = graph.inputs
        mlp = MLP(Linear(w1), Linear(w2), Linear(w3))
        graph.output(mlp(x))
        return graph


@pytest.mark.parametrize(
    "input_type",
    [
        TensorType(DType.float32, ["dim"], DeviceRef.GPU(id=0)),
        TensorType(DType.float32, ["batch", "dim"], DeviceRef.GPU(id=0)),
    ],
)
def test_mlp(input_type: TensorType):
    # Get Graph
    host = CPU(0)
    device0 = Accelerator(0)
    session = InferenceSession(devices=[device0])

    dim = input_type.shape[-1]
    w1_type: TensorType = TensorType(
        input_type.dtype, ["hidden_dim", dim], DeviceRef.GPU(id=0)
    )
    w2_type: TensorType = TensorType(
        input_type.dtype, [dim, "hidden_dim"], DeviceRef.GPU(id=0)
    )
    w3_type: TensorType = w1_type

    graph = mlp_graph([input_type, w1_type, w2_type, w3_type])
    compiled = session.load(graph)
    if input_type.rank == 1:
        x_np = np.ones((128)).astype(np.float32)
    else:
        x_np = np.ones((32, 128)).astype(np.float32)
    w1_np = np.ones((16, 128)).astype(np.float32)
    w2_np = np.ones((128, 16)).astype(np.float32)
    w3_np = np.ones((16, 128)).astype(np.float32)

    x = Tensor.from_numpy(x_np)
    w1 = Tensor.from_numpy(w1_np)
    w2 = Tensor.from_numpy(w2_np)
    w3 = Tensor.from_numpy(w3_np)

    results = compiled.execute(x, w1, w2, w3)
    expected = (
        TorchMLP(torch.tensor(w1_np), torch.tensor(w2_np), torch.tensor(w3_np))(
            torch.tensor(x_np)
        )
        .detach()
        .numpy()
    )
    ACCURACY_RTOL = 1e-1
    ACCURACY_ATOL = 1e-6
    for result in results:
        assert isinstance(result, Tensor)
        np.testing.assert_allclose(
            result.to(host).to_numpy(),
            expected,
            atol=ACCURACY_ATOL,
            rtol=ACCURACY_RTOL,
            equal_nan=True,
        )
