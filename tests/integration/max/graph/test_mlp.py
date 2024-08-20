# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import hypothesis
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hypothesis import strategies as st
from hypothesis.extra import numpy as nps
from hypothesis.strategies import integers, lists, shared, tuples
from llama3.mlp import MLP, Linear
from max.graph import DType, Graph, TensorType


class TorchMLP(nn.Module):
    def __init__(self, w1, w2, w3):
        super().__init__()
        self.gate_proj = nn.Linear(w1.shape[0], w1.shape[1], bias=False)
        self.gate_proj.weight = nn.Parameter(w1)
        self.down_proj = nn.Linear(w2.shape[0], w2.shape[1], bias=False)
        self.down_proj.weight = nn.Parameter(w2)
        self.up_proj = nn.Linear(w3.shape[0], w3.shape[1], bias=False)
        self.up_proj.weight = nn.Parameter(w3)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# GRA-855, Currently batched matmul is limited to tensors with rank less than 4
header = shared(
    lists(integers(min_value=1, max_value=4), min_size=3, max_size=4),
    key="shared",
)

hidden_state_size = shared(integers(min_value=1, max_value=6), key="hidden")
intermediate_state_size = shared(
    integers(min_value=1, max_value=6), key="intermediate"
)
output_size = shared(integers(min_value=1, max_value=6), key="output_size")


@hypothesis.given(
    x=nps.arrays(
        dtype=np.float32,
        shape=tuples(
            header, lists(hidden_state_size, min_size=1, max_size=1)
        ).map(lambda x: x[0] + x[1]),
    ),
    w1=nps.arrays(
        dtype=np.float32,
        shape=tuples(hidden_state_size, intermediate_state_size),
    ),
    w2=nps.arrays(
        dtype=np.float32,
        shape=tuples(intermediate_state_size, output_size),
    ),
    w3=nps.arrays(
        dtype=np.float32,
        shape=tuples(hidden_state_size, intermediate_state_size),
    ),
)
@hypothesis.settings(max_examples=16, deadline=None)
def test_mlp(session, x, w1, w2, w3):
    # Initialize Max MLP
    graph = Graph(
        "mlp",
        MLP(Linear(w1), Linear(w2), Linear(w3)),
        input_types=[TensorType(dtype=DType.float32, shape=x.shape)],
    )

    compiled = session.load(graph)

    # Initialize Pytorch MLP
    torch_mlp = TorchMLP(*[torch.from_numpy(v) for v in [w1.T, w2.T, w3.T]])

    # Execute two options
    generated = compiled.execute(input0=x)
    expected = torch_mlp(torch.from_numpy(x)).detach().numpy()
    np.testing.assert_almost_equal(generated["output0"], expected, decimal=4)
