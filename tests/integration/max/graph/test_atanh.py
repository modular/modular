# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import platform

import numpy as np
import pytest
import torch
from max.driver import Tensor
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops


@pytest.mark.parametrize("dtype", [DType.float32, DType.bfloat16])
def test_atanh(session, dtype):
    if dtype == DType.bfloat16 and platform.machine() in ["arm64", "aarch64"]:
        pytest.skip("BF16 is not supported on ARM CPU architecture")

    input_type = TensorType(dtype, [1024], device=DeviceRef.CPU())

    with Graph(f"atanh_{dtype}", input_types=[input_type]) as graph:
        out = ops.atanh(graph.inputs[0])
        graph.output(out.cast(DType.float32))

    model = session.load(graph)

    # Set fixed seed for reproducibility
    torch.manual_seed(42)

    torch_dtype = torch.float32 if dtype == DType.float32 else torch.bfloat16

    # Generate random values between [-1, 1] with a small epsilon,
    # domain of atanh is [-1, 1]
    extreme_value = 1.0 - 1e-3
    input_data = torch.rand(1024, dtype=torch_dtype) * 2.0 - 1.0
    input_data = torch.clamp(input_data, min=-extreme_value, max=extreme_value)

    max_result = model(
        Tensor.from_dlpack(input_data).to(model.input_devices[0])
    )[0]
    max_result = max_result.to_numpy()

    torch_result = torch.atanh(input_data).to(dtype=torch.float32).cpu().numpy()

    np.testing.assert_allclose(
        max_result,
        torch_result,
        rtol=1e-1,
        atol=1e-1,
        verbose=True,
    )
