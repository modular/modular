# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import numpy as np
import pytest
import torch
from max.driver import Tensor
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops


@pytest.mark.parametrize(
    "input_shape, kernel_size, stride, ceil_mode",
    [
        ((1, 6, 15, 1), (3, 2), (3, 2), False),
        ((1, 6, 15, 1), (3, 2), (1, 1), False),
        ((1, 6, 15, 1), (3, 2), (3, 2), True),
    ],
)
def test_max_pool(session, input_shape, kernel_size, stride, ceil_mode):
    device_ref = DeviceRef.from_device(session.devices[0])
    torch_device = "cpu"

    dilation = (1, 1)
    padding = (0, 0)

    input_tensor = torch.randn(
        input_shape, device=torch_device, dtype=torch.float32
    )
    expected = torch.nn.functional.max_pool2d(
        input_tensor.permute(0, 3, 1, 2),  # NHWC -> NCHW
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode=ceil_mode,
    )
    expected = expected.permute(0, 2, 3, 1)  # NCHW -> NHWC
    with Graph(
        "max_pool2d",
        input_types=(
            TensorType(
                DType.float32, ["batch", "x", "y", "channel"], device_ref
            ),
        ),
    ) as graph:
        output = ops.max_pool2d(
            graph.inputs[0].tensor,
            kernel_size,
            stride,
            dilation,
            padding,
            ceil_mode=ceil_mode,
        )
        graph.output(output)

    model = session.load(graph)
    max_input = Tensor.from_numpy(input_tensor.numpy()).to(session.devices[0])
    model_output = model(max_input)[0]
    assert isinstance(model_output, Tensor)
    actual = model_output.to_numpy()
    np.testing.assert_equal(actual, expected.numpy(force=True))


@pytest.mark.parametrize(
    "input_shape, kernel_size, stride, padding, ceil_mode, count_boundary",
    [
        ((1, 6, 15, 1), (3, 2), (3, 2), (0, 0), False, True),
        ((1, 6, 15, 1), (3, 2), (3, 2), (1, 1), False, True),
    ],
)
def test_avg_pool(
    session,
    input_shape,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_boundary,
):
    device_ref = DeviceRef.from_device(session.devices[0])
    torch_device = "cpu"

    dilation = (1, 1)

    input_tensor = torch.randn(
        input_shape, device=torch_device, dtype=torch.float32
    )
    expected = torch.nn.functional.avg_pool2d(
        input_tensor.permute(0, 3, 1, 2),  # NHWC -> NCHW
        kernel_size,
        stride,
        padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_boundary,
    )
    expected = expected.permute(0, 2, 3, 1)  # NCHW -> NHWC
    with Graph(
        "avg_pool2d",
        input_types=(
            TensorType(
                DType.float32, ["batch", "x", "y", "channel"], device_ref
            ),
        ),
    ) as graph:
        output = ops.avg_pool2d(
            graph.inputs[0].tensor,
            kernel_size,
            stride,
            dilation,
            padding,
            ceil_mode=ceil_mode,
            count_boundary=count_boundary,
        )
        graph.output(output)

    model = session.load(graph)
    avg_input = Tensor.from_numpy(input_tensor.numpy()).to(session.devices[0])
    model_output = model(avg_input)[0]
    assert isinstance(model_output, Tensor)
    actual = model_output.to_numpy()
    np.testing.assert_almost_equal(actual, expected.numpy(force=True))
