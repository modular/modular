# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Runs Llama3.2 vision MLP and compares it against the transformers package
reference implementation.
"""

import numpy as np
import pytest

from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType
from modular_graph_test import modular_graph_test
from nn import Linear
from torch_utils import TorchVisionEncoderMLP
from llama3.vision.mlp import MLP


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
def test_mlp(session: InferenceSession, input_type: TensorType) -> None:
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
            expected = TorchVisionEncoderMLP(w1, w2)(x).detach().numpy()
            # Relative L2 norm threshold
            threshold = 0.01
            assert (
                np.linalg.norm(result - expected)
                / (np.linalg.norm(expected) + np.finfo(np.float32).eps)
                < threshold
            )
