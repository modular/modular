# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from datetime import timedelta

import hypothesis
import hypothesis.strategies as st
import max.engine as me
import numpy as np
import torch
import torch.nn as nn
from hypothesis.extra.numpy import arrays
from llama3.norm import RMSNorm
from max.graph import DType, Graph, TensorType


class TorchRMSNorm(torch.nn.Module):
    def __init__(self, weight: float, eps: float = 1e-6):
        """
        This is sourced from: https://github.com/meta-llama/llama/blob/main/llama/model.py#L34
        """
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


header = st.shared(
    st.lists(st.integers(min_value=5, max_value=10), min_size=2, max_size=5),
    key="header",
)
hidden_size = st.shared(
    st.lists(st.integers(min_value=5, max_value=10), min_size=1, max_size=1),
    key="hidden_size",
)
input_shape = st.tuples(header, hidden_size).map(lambda x: x[0] + x[1])


@hypothesis.given(
    st.floats(min_value=1e-6, max_value=1),
    arrays(
        np.float32,
        elements=st.floats(
            min_value=9.999999974752427e-07, max_value=2.0, width=32
        ),
        shape=hidden_size,
    ),
    arrays(
        np.float32,
        elements=st.floats(
            min_value=9.999999974752427e-07, max_value=2.0, width=32
        ),
        shape=input_shape,
    ),
)
@hypothesis.settings(max_examples=10, deadline=timedelta(seconds=3))
def test_norm(eps, weights, input):
    # Initialize Graph
    # TODO: This is not resulting in nan, when all input values are 0
    session = me.InferenceSession()
    norm = Graph(
        "norm",
        RMSNorm(weight=weights, eps=np.array(eps)),
        input_types=[TensorType(dtype=DType.float32, shape=input.shape)],
    )
    compiled = session.load(norm)

    # Initialize Pytorch RMSNorm
    torch_norm = TorchRMSNorm(
        eps=torch.tensor(eps), weight=torch.from_numpy(weights)
    )

    # Execute two options
    generated = compiled.execute(input0=input)
    expected = torch_norm(torch.from_numpy(input)).detach().numpy()
    np.testing.assert_almost_equal(generated["output0"], expected, decimal=4)
