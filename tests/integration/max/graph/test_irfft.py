# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import max.driver as md
import pytest
import torch
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops


def max_irfft(session, input_tensor, n, axis, normalization):
    with Graph(
        "irfft",
        input_types=(
            TensorType(DType.float32, input_tensor.shape, DeviceRef.GPU()),
        ),
    ) as graph:
        output = ops.irfft(graph.inputs[0], n, axis, normalization)
        graph.output(output)
    model = session.load(graph)
    output = model(input_tensor)
    output = torch.from_dlpack(output[0])
    return output


def torch_irfft(input_tensor, n, axis, normalization):
    output = torch.fft.irfft(input_tensor, n=n, dim=axis, norm=normalization)
    return output


@pytest.mark.parametrize(
    "input_shape,n,axis,normalization",
    [
        ((5, 10, 15), 3, -1, "backward"),
        ((5, 10, 15), 20, 0, "ortho"),
        ((5, 10, 15), None, 1, "forward"),
    ],
)
def test_irfft(session, input_shape, n, axis, normalization) -> None:
    if md.accelerator_count() == 0:
        pytest.skip("No GPU available")
    if md.accelerator_api() != "cuda":
        pytest.skip("NVIDIA GPUs are required for this test.")
    input_tensor = torch.randn(*input_shape, dtype=torch.float32).to("cuda")
    max_out = max_irfft(session, input_tensor, n, axis, normalization)
    torch_out = torch_irfft(input_tensor, n, axis, normalization)

    torch.testing.assert_close(
        torch_out,
        max_out,
        rtol=1e-6,
        atol=2 * torch.finfo(torch.float32).eps,
    )
