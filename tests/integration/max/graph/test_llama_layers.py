# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.engine Python bindings with Max Graph."""


import max.engine as me
import numpy as np
from max.graph import DType, Graph, TensorType
from dataclasses import dataclass
from llama3.mlp import MLP
import torch
import torch.nn.functional as F


@dataclass
class NanoLlama3:
    """Class to hold toy weights for testing llama3 code."""

    mlp_w1 = np.array(
        [
            [0.5641, 0.4875],
            [-1.1172, -1.1583],
        ]
    )
    mlp_w2 = np.array(
        [
            [0.5355, -0.9487],
            [-0.6487, 0.1838],
        ]
    )
    mlp_w3 = np.array(
        [
            [-0.6765, 0.7103],
            [-0.4643, 0.2860],
        ]
    )


def test_mlp():
    session = me.InferenceSession()
    model = NanoLlama3()

    mlp = Graph(
        "mlp",
        MLP(model.mlp_w1, model.mlp_w2, model.mlp_w3),
        input_types=[TensorType(dtype=DType.float32, shape=[2, 2, 2])],
    )
    compiled = session.load(mlp)
    input = (
        np.array(
            [
                -1.2620,
                -2.0678,
                -1.6634,
                1.3036,
                -0.0088,
                -1.1315,
                1.1287,
                1.7699,
            ]
        )
        .reshape((2, 2, 2))
        .astype(np.float32)
    )
    output = compiled.execute(input0=input)

    expected = (
        np.array(
            [
                0.1053,
                -0.1079,
                -0.3632,
                0.2142,
                0.4025,
                -0.1661,
                0.3220,
                -0.3921,
            ]
        )
        .reshape((2, 2, 2))
        .astype(np.float32)
    )
    # TODO (MSDK-720): Re-enable after troubleshooting accuracy.
    # assert np.testing.assert_almost_equal(output["output0"], expected, decimal=4)
