# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# This code is shared between test_conv.py and test_conv_gpu.py


import numpy as np
import torch
import torch.nn as nn
from max.driver import Tensor
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn import Conv1D, Conv3D

ACCURACY_RTOL = 2e-4
ACCURACY_ATOL = 1e-5


def conv3d_impl(session: InferenceSession) -> None:
    torch.manual_seed(42)

    in_channels = 3
    out_channels = 1280
    kernel_size = (2, 14, 14)
    stride = (2, 14, 14)

    # input params
    batch_size = 3
    depth = 32
    height = 112
    width = 112

    is_gpu = not session.devices[0].is_host
    torch_dtype = torch.float32
    torch_device = torch.device("cuda") if is_gpu else torch.device("cpu")
    max_dtype = DType.float32
    max_device = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()

    input_sequence = torch.rand(
        size=(batch_size, in_channels, depth, height, width),
        dtype=torch_dtype,
        device=torch_device,
    )

    torch_conv = nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        bias=False,
        device=torch_device,
    )

    max_conv = Conv3D(
        depth=kernel_size[0],
        height=kernel_size[1],
        width=kernel_size[2],
        in_channels=in_channels,
        out_channels=out_channels,
        dtype=max_dtype,
        stride=stride,
        has_bias=False,
        permute=True,
        device=max_device,
    )

    # load random weights to torch
    torch_conv.weight.data = nn.Parameter(
        torch.rand(size=torch_conv.weight.data.shape, device=torch_device)
    )

    # load weights to max
    state_dict = {"weight": torch_conv.weight.data.detach().cpu()}
    max_conv.load_state_dict(state_dict)

    # get_torch_output
    with torch.no_grad():
        torch_conv_result = torch_conv(input_sequence)

    # get_max_output
    graph = Graph(
        "conv3d",
        max_conv,
        input_types=(
            TensorType(max_dtype, input_sequence.shape, device=max_device),
        ),
    )

    compiled = session.load(graph, weights_registry=max_conv.state_dict())

    graph_api_conv_result = compiled.execute(input_sequence)[0]
    assert isinstance(graph_api_conv_result, Tensor)

    np.testing.assert_allclose(
        graph_api_conv_result.to_numpy(),
        torch_conv_result.detach().cpu().numpy(),
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )


def conv1d_impl(session: InferenceSession) -> None:
    torch.manual_seed(42)

    batch_size = 1
    in_channels = 1024
    length = 57
    hidden_size = 1024  # out_channels
    kernel_size = 7
    stride = 1
    padding = 3

    is_gpu = not session.devices[0].is_host
    torch_dtype = torch.float32
    torch_device = torch.device("cuda") if is_gpu else torch.device("cpu")
    max_dtype = DType.float32
    max_device = DeviceRef.GPU() if is_gpu else DeviceRef.CPU()

    # Create input tensor in PyTorch format (batch_size, in_channels, length)
    input_sequence = torch.rand(
        size=(batch_size, in_channels, length),
        dtype=torch_dtype,
        device=torch_device,
    )

    # Create PyTorch Conv1d layer
    torch_conv = nn.Conv1d(
        in_channels=in_channels,
        out_channels=hidden_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
        device=torch_device,
    )

    # Create our Conv1D layer
    max_conv = Conv1D(
        kernel_size=kernel_size,
        in_channels=in_channels,
        out_channels=hidden_size,
        dtype=max_dtype,
        stride=stride,
        device=max_device,
        padding=padding,
        has_bias=True,
        permute=True,
    )

    # Initialize random weights for PyTorch conv
    torch_conv.weight.data = nn.Parameter(
        torch.rand(
            size=torch_conv.weight.data.shape,
            dtype=torch_dtype,
            device=torch_device,
        )
    )
    assert torch_conv.bias is not None

    torch_conv.bias.data = nn.Parameter(
        torch.rand(
            size=torch_conv.bias.data.shape,
            dtype=torch_dtype,
            device=torch_device,
        )
    )
    # Load the same weights into our conv
    state_dict = {
        "weight": torch_conv.weight.data.detach().cpu(),
        "bias": torch_conv.bias.data.detach().cpu(),
    }
    max_conv.load_state_dict(state_dict)

    # Get PyTorch output
    with torch.no_grad():
        torch_conv_result = torch_conv(input_sequence)

    # Get Max output
    graph = Graph(
        "conv1d",
        max_conv,
        input_types=(
            TensorType(max_dtype, input_sequence.shape, device=max_device),
        ),
    )

    compiled = session.load(graph, weights_registry=max_conv.state_dict())
    graph_api_conv_result = compiled.execute(input_sequence)[0]
    assert isinstance(graph_api_conv_result, Tensor)

    np.testing.assert_allclose(
        graph_api_conv_result.to_numpy(),
        torch_conv_result.detach().cpu().numpy(),
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )
