# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the is_inf operator."""

import platform

import numpy as np
import pytest
import torch
from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


@pytest.mark.parametrize("dtype", [DType.float32, DType.bfloat16])
def test_is_inf(session: InferenceSession, dtype) -> None:
    if dtype == DType.bfloat16 and platform.machine() in ["arm64", "aarch64"]:
        pytest.skip("BF16 is not supported on ARM CPU architecture")

    input_type = TensorType(dtype, [8], device=DeviceRef.CPU())

    with Graph(f"is_inf_{dtype}", input_types=[input_type]) as graph:
        out = ops.is_inf(graph.inputs[0].tensor)
        graph.output(out)

    model = session.load(graph)

    torch_dtype = torch.float32 if dtype == DType.float32 else torch.bfloat16

    # Create test data with some infinity values
    input_data = torch.tensor(
        [
            1.0,
            float("inf"),
            3.0,
            float("nan"),
            -float("inf"),
            0.0,
            -2.5,
            float("inf"),
        ],
        dtype=torch_dtype,
    )

    max_result = model(input_data)[0]
    assert isinstance(max_result, Tensor)
    max_result_np = max_result.to_numpy()

    # Compare with NumPy's isinf
    torch_result = torch.isinf(input_data).cpu().numpy()

    np.testing.assert_array_equal(
        max_result_np,
        torch_result,
        verbose=True,
    )
