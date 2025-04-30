# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import numpy as np
import pytest
import torch
from max import nn
from max.driver import Tensor, accelerator_count
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType

device_ref = DeviceRef.GPU() if accelerator_count() > 0 else DeviceRef.CPU()


@pytest.mark.parametrize("dtype", [DType.float32, DType.int16])
def test_clamp(session, dtype):
    input_type = TensorType(dtype, [10, 10], device=device_ref)

    with Graph(f"clamp_{dtype}", input_types=[input_type]) as graph:
        out = nn.clamp(graph.inputs[0], min=10, max=20)
        graph.output(out.cast(DType.float32))

    model = session.load(graph)

    torch_dtype = torch.float32 if dtype == DType.float32 else torch.int16
    input_data = torch.arange(end=100, dtype=torch_dtype).reshape((10, 10))

    max_result = model(
        Tensor.from_dlpack(input_data).to(model.input_devices[0])
    )[0]
    max_result = max_result.to_numpy()

    torch_result = (
        torch.clamp(input_data, min=10, max=20)
        .to(dtype=torch.float32)
        .cpu()
        .numpy()
    )

    np.testing.assert_allclose(
        max_result,
        torch_result,
        rtol=1e-6,
        atol=1e-6,
        verbose=True,
    )
