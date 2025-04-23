# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import pytest
import torch
from max.driver import accelerator_api
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn import RMSNorm
from modular_graph_test import are_all_tensor_values, modular_graph_test


def torch_rms_norm(x, weight, eps=1e-6):
    #   See https://github.com/meta-llama/llama/blob/main/llama/model.py#L34
    return x * torch.rsqrt((x**2).mean(-1, keepdim=True) + eps) * weight


def run_test_norm(
    session: InferenceSession,
    input_type: TensorType,
    rtol: float,
    atol: float,
):
    # Initialize Graph
    dim = input_type.shape[-1]
    weight_type = TensorType(input_type.dtype, [dim], device=input_type.device)
    with Graph("norm", input_types=[input_type, weight_type]) as graph:
        assert are_all_tensor_values(graph.inputs)
        x, weight = graph.inputs
        graph.output(RMSNorm(weight)(x))

        @modular_graph_test(session, graph)
        def test_correctness(execute, inputs, torch_inputs):
            result = torch.from_dlpack(execute(inputs))
            expected = torch_rms_norm(*torch_inputs)
            torch.testing.assert_close(
                result,
                expected,
                atol=atol,
                rtol=rtol,
                equal_nan=True,
                check_device=False,
                msg=lambda msg: f"\n{result=}\n{expected=}\n",
            )


# ===----------------------------------------------------------------------=== #
# CPU Test
# ===----------------------------------------------------------------------=== #


SHAPES = (
    ["dim"],
    ["batch", "dim"],
    ["a", "x", "y", "z", "dim"],
)

CPU_DTYPES = (DType.float32, DType.float64)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", CPU_DTYPES)
def test_norm(session, shape, dtype):
    run_test_norm(
        session,
        TensorType(dtype, shape, device=DeviceRef.CPU()),
        rtol=1e-2,
        atol=1e-8,
    )


# ===----------------------------------------------------------------------=== #
# GPU Test
# ===----------------------------------------------------------------------=== #


# TODO(MAXPLAT-118): float64 is broken for GPU
# GPU_DTYPES = (*CPU_DTYPES, DType.bfloat16)
GPU_DTYPES = (DType.float32, DType.bfloat16)


@pytest.mark.skipif(accelerator_api() == "cpu", reason="Test only runs on GPU")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", GPU_DTYPES)
def test_norm_gpu(gpu_session, shape, dtype):
    run_test_norm(
        gpu_session,
        TensorType(dtype, shape, device=DeviceRef.GPU()),
        rtol=1e-1,
        atol=1e-8,
    )
