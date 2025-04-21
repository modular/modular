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
from max.graph import Graph, TensorType, ops


@pytest.mark.parametrize("dtype", [DType.float32, DType.bfloat16])
def test_cumsum(session, dtype):
    if dtype == DType.bfloat16 and platform.machine() in ["arm64", "aarch64"]:
        pytest.skip("BF16 is not supported on ARM CPU architecture")

    input_type = TensorType(dtype, [1024])

    with Graph(f"cumsum_{dtype}", input_types=[input_type]) as graph:
        out = ops.cumsum(graph.inputs[0], axis=0)
        graph.output(out.cast(DType.float32))

    model = session.load(graph)

    torch_dtype = torch.float32 if dtype == DType.float32 else torch.bfloat16
    input_data = torch.full((1024,), 1.1, dtype=torch_dtype)

    max_result = model(
        Tensor.from_dlpack(input_data).to(model.input_devices[0])
    )[0]
    max_result = max_result.to_numpy()

    torch_result = (
        torch.cumsum(input_data, dim=0).to(dtype=torch.float32).cpu().numpy()
    )

    np.testing.assert_allclose(
        max_result,
        torch_result,
        rtol=1e-6,
        atol=1e-6,
        verbose=True,
    )
