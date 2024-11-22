# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import numpy as np
import pytest
import torch.nn.functional as F
from hypothesis import example
from max.dtype import DType
from max.graph import Graph, TensorType
from modular_graph_test import modular_graph_test
from nn import LPLayerNorm


def torch_layer_norm(x, weight):
    return F.layer_norm(x, weight.shape, weight=weight, eps=1e-6)


@pytest.mark.xfail(True, reason="AIPIPE-153")
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
        graph.output(LPLayerNorm(weight)(x))  # type: ignore

        @example(
            (
                np.array([0.0, 1.0], dtype=np.float32),
                np.array([1.0, 0.0], dtype=np.float32),
            )
        )
        @modular_graph_test(session, graph)
        def test_correctness(execute, inputs, torch_inputs):
            result = execute(inputs)
            expected = torch_layer_norm(*torch_inputs).detach().numpy()
            assert result.shape == expected.shape
            # In some cases Torch produces NaN while we produce zero.  Fudge
            # the Torch results to zero for locations where it produces NaN and
            # we don't.
            expected[np.isnan(expected) & ~np.isnan(result)] = 0
            ACCURACY_RTOL = 1e-2
            ACCURACY_ATOL = 1e-8
            np.testing.assert_allclose(
                result,
                expected,
                atol=ACCURACY_ATOL,
                rtol=ACCURACY_RTOL,
                equal_nan=True,
            )
