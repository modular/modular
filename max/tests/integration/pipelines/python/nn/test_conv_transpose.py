# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import numpy as np
import torch
import torch.nn as nn
from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn import ConvTranspose1d

ACCURACY_RTOL = 1e-4
ACCURACY_ATOL = 1e-6


def test_conv_transpose1d() -> None:
    batch_size = 10
    in_channels = 16
    length = 3
    out_channels = 33  # out_channels. Has nothing to do with input size
    kernel_size = 5
    stride = 2
    padding = 3
    dilation = 1
    output_padding = 1

    is_gpu = False
    torch_dtype = torch.float32
    max_dtype = DType.float32

    # batch_size, in_channels, seq_length = 3000
    input_sequence = torch.rand(size=(batch_size, in_channels, length)).to(
        torch_dtype
    )
    input_sequence = input_sequence.cuda() if is_gpu else input_sequence.cpu()

    torch_conv = nn.ConvTranspose1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_padding=output_padding,
        bias=False,
    )

    max_conv = ConvTranspose1d(
        length=kernel_size,
        in_channels=in_channels,
        out_channels=out_channels,
        dtype=max_dtype,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_padding=output_padding,
        permute=True,
    )

    # load random weights to torch
    torch_conv.weight.data = nn.Parameter(
        torch.rand(size=torch_conv.weight.data.shape)
    )

    # load weights to max
    state_dict = {"weight": torch_conv.weight.data.cpu()}
    max_conv.load_state_dict(state_dict)

    # get_torch_output
    with torch.no_grad():
        torch_conv_result = torch_conv(input_sequence)

    # get_max_output
    session = InferenceSession()
    graph = Graph(
        "conv_transpose1d",
        max_conv,
        input_types=(
            TensorType(max_dtype, input_sequence.shape, DeviceRef.CPU()),
        ),
    )

    compiled = session.load(graph, weights_registry=max_conv.state_dict())

    max_conv_result = compiled.execute(input_sequence)[0]
    assert isinstance(max_conv_result, Tensor)

    np.testing.assert_allclose(
        max_conv_result.to_numpy(),
        torch_conv_result.detach().numpy(),
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )


def test_conv_transpose1d_bias() -> None:
    batch_size = 10
    in_channels = 16
    length = 3
    out_channels = 33  # out_channels. Has nothing to do with input size
    kernel_size = 5
    stride = 2
    padding = 3
    dilation = 1
    output_padding = 1

    is_gpu = False
    torch_dtype = torch.float32
    max_dtype = DType.float32

    # batch_size, in_channels, seq_length = 3000
    input_sequence = torch.rand(size=(batch_size, in_channels, length)).to(
        torch_dtype
    )
    input_sequence = input_sequence.cuda() if is_gpu else input_sequence.cpu()

    torch_conv = nn.ConvTranspose1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_padding=output_padding,
        bias=True,
    )

    max_conv = ConvTranspose1d(
        length=kernel_size,
        in_channels=in_channels,
        out_channels=out_channels,
        dtype=max_dtype,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_padding=output_padding,
        permute=True,
        has_bias=True,
    )

    # load random weights to torch
    torch_conv.weight.data = nn.Parameter(
        torch.rand(size=torch_conv.weight.data.shape)
    )

    # load weights to max
    state_dict = {"weight": torch_conv.weight.data.cpu()}
    state_dict.update({"bias": torch_conv.bias.data.cpu()})

    max_conv.load_state_dict(state_dict)

    # get_torch_output
    with torch.no_grad():
        torch_conv_result = torch_conv(input_sequence)

    # get_max_output
    session = InferenceSession()
    graph = Graph(
        "conv_transpose1d",
        max_conv,
        input_types=(
            TensorType(max_dtype, input_sequence.shape, DeviceRef.CPU()),
        ),
    )

    compiled = session.load(graph, weights_registry=max_conv.state_dict())

    max_conv_result = compiled.execute(input_sequence)[0]
    assert isinstance(max_conv_result, Tensor)

    np.testing.assert_allclose(
        max_conv_result.to_numpy(),
        torch_conv_result.detach().numpy(),
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )
