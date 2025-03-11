# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import numpy as np
import pytest
import torch
from max.driver import accelerator_api
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType
from max.nn import RMSNorm
from modular_graph_test import are_all_tensor_values, modular_graph_test


def torch_rms_norm(x, weight, eps=1e-6):
    #   See https://github.com/meta-llama/llama/blob/main/llama/model.py#L34
    return x * torch.rsqrt((x**2).mean(-1, keepdim=True) + eps) * weight


def run_test_norm(session: InferenceSession, input_type: TensorType):
    # Initialize Graph
    dim = input_type.shape[-1]
    weight_type = TensorType(input_type.dtype, [dim])
    with Graph("norm", input_types=[input_type, weight_type]) as graph:
        assert are_all_tensor_values(graph.inputs)
        x, weight = graph.inputs
        graph.output(RMSNorm(weight)(x))

        @modular_graph_test(session, graph)
        def test_correctness(execute, inputs, torch_inputs):
            result = execute(inputs)
            expected = torch_rms_norm(*torch_inputs).detach().numpy()
            ACCURACY_RTOL = 1e-2
            ACCURACY_ATOL = 1e-8
            np.testing.assert_allclose(
                result,
                expected,
                atol=ACCURACY_ATOL,
                rtol=ACCURACY_RTOL,
                equal_nan=True,
            )


# ===----------------------------------------------------------------------=== #
# CPU Test
# ===----------------------------------------------------------------------=== #


def get_tensor_types(type: DType):
    return [
        TensorType(type, ["dim"]),
        TensorType(type, ["batch", "dim"]),
        TensorType(type, ["a", "x", "y", "z", "dim"]),
        TensorType(type, ["dim"]),
    ]


@pytest.mark.parametrize(
    "input_type",
    [
        *get_tensor_types(DType.float32),
        *get_tensor_types(DType.float64),
    ],
)
def test_norm(session, input_type):
    run_test_norm(session, input_type)


# ===----------------------------------------------------------------------=== #
# GPU Test
# ===----------------------------------------------------------------------=== #


@pytest.mark.skipif(accelerator_api() == "cpu", reason="Test only runs on GPU")
@pytest.mark.parametrize(
    "input_type",
    [
        *get_tensor_types(DType.bfloat16),
        *get_tensor_types(DType.float32),
        *get_tensor_types(DType.float64),
    ],
)
def test_norm_gpu(gpu_session, input_type):
    run_test_norm(gpu_session, input_type)
