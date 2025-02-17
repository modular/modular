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
from max.graph import Graph, TensorType, Weight
from max.pipelines.nn import Conv1D, Conv3D

ACCURACY_RTOL = 1e-4
ACCURACY_ATOL = 1e-6


def test_conv1d() -> None:
    batch_size = 3
    in_channels = 128
    length = 3000
    hidden_size = 1280  # out_channels. Has nothing to do with input size
    kernel_size = 3
    stride = 1  # try with 2
    padding = 1

    # batch_size, in_channels, seq_length = 3000
    input_sequence = torch.rand(size=(batch_size, in_channels, length)).to(
        torch.float32
    )

    conv = nn.Conv1d(
        in_channels=in_channels,
        out_channels=hidden_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )

    conv.weight.data = nn.Parameter(torch.rand(size=conv.weight.data.shape))
    conv.bias.data = nn.Parameter(torch.rand(size=conv.bias.data.shape))

    # out_length = ((length + 2 * padding - (kernel_size - 1) - 1) / stride) + 1
    # out_length = ((3000 + 2 * 1 - (3 - 1) - 1) / 1) + 1 = 3000
    with torch.no_grad():
        # permute output from (batch_size, hidden_size, out_length) to (batch_size, out_length, hidden_size).
        torch_conv_result = torch.permute(conv(input_sequence), (0, 2, 1))

    # Permute (batch_size, n_channels, seq_length) inputs to (batch_size, seq_length, n_channels) for our Graph API.
    graph_api_inputs = torch.permute(input_sequence, (0, 2, 1)).contiguous()
    # Permute filters from (hidden_size, in_channels, kernel_size) to (kernel_size, in_channels, hidden_size).
    weights_registry = {}
    weights_registry["conv1d_weight"] = torch.permute(
        conv.weight.data, (2, 1, 0)
    ).contiguous()
    weights_registry["conv1d_bias"] = conv.bias.data.contiguous()

    graph_api_filters = Weight(
        name="conv1d_weight",
        dtype=DType.from_numpy(weights_registry["conv1d_weight"].numpy().dtype),
        shape=weights_registry["conv1d_weight"].shape,
    )
    graph_api_bias = Weight(
        name="conv1d_bias",
        dtype=DType.from_numpy(weights_registry["conv1d_bias"].numpy().dtype),
        shape=weights_registry["conv1d_bias"].shape,
    )

    # out_channels=hidden_size and kernel_size=kernel_size are inferred from kernel.
    session = InferenceSession()
    graph = Graph(
        "conv1d",
        Conv1D(
            graph_api_filters,
            bias=graph_api_bias,
            stride=stride,
            padding=padding,
        ),
        input_types=(
            TensorType(DType.float32, (batch_size, length, in_channels)),
        ),
    )

    compiled = session.load(graph, weights_registry=weights_registry)

    graph_api_conv_result = compiled.execute(graph_api_inputs)[0]
    assert isinstance(graph_api_conv_result, Tensor)

    np.testing.assert_allclose(
        graph_api_conv_result.to_numpy(),
        torch_conv_result.detach().numpy(),
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )


def test_conv3d() -> None:
    in_channels = 3
    out_channels = 1280
    kernel_size = (2, 14, 14)
    stride = (2, 14, 14)

    # input params
    batch_size = 3
    depth = 32
    height = 112
    width = 112

    # batch_size, in_channels, depth, height, width
    input_sequence = torch.rand(
        size=(batch_size, in_channels, depth, height, width)
    ).to(torch.float32)

    conv = nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        bias=False,
    )

    conv.weight.data = nn.Parameter(torch.rand(size=conv.weight.data.shape))
    print("Filter shape = ", conv.weight.data.shape)
    print(
        "Should be = (out_channels, in_channels, depth, height, width) = ",
        (out_channels, in_channels, depth, height, width),
    )

    with torch.no_grad():
        # permute output from (batch_size, out_channels, depth, height, width) to (batch_size, depth, height, width, out_channels).
        torch_conv_result = torch.permute(conv(input_sequence), (0, 2, 3, 4, 1))

    # Permute (batch_size, in_channels, depth, height, width) inputs to (batch_size, depth, height, width, in_channels) for our Graph API.
    graph_api_inputs = torch.permute(
        input_sequence, (0, 2, 3, 4, 1)
    ).contiguous()
    # Permute filters from (out_channels, in_channels, depth, height, width) to (depth, height, width, in_channels / num_groups, out_channels).
    weights_registry = {}
    weights_registry["conv3d_weight"] = torch.permute(
        conv.weight.data, (2, 3, 4, 1, 0)
    ).contiguous()

    graph_api_filters = Weight(
        name="conv3d_weight",
        dtype=DType.from_numpy(weights_registry["conv3d_weight"].numpy().dtype),
        shape=weights_registry["conv3d_weight"].shape,
    )

    session = InferenceSession()
    graph = Graph(
        "conv3d",
        Conv3D(
            graph_api_filters,
            bias=None,
            stride=stride,
        ),
        input_types=(
            TensorType(
                DType.float32, (batch_size, depth, height, width, in_channels)
            ),
        ),
    )

    compiled = session.load(graph, weights_registry=weights_registry)

    graph_api_conv_result = compiled.execute(graph_api_inputs)[0]
    assert isinstance(graph_api_conv_result, Tensor)

    np.testing.assert_allclose(
        graph_api_conv_result.to_numpy(),
        torch_conv_result.detach().numpy(),
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )
