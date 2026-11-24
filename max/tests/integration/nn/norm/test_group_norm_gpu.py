# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import pytest
import torch
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.nn.legacy import GroupNorm


def load_max_group_norm(
    gpu_session: InferenceSession,
    shape: torch.Size,
    num_groups: int,
    num_channels: int,
    eps: float = 1e-5,
    affine: bool = False,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> Model:
    device_ref = DeviceRef.GPU()
    with Graph(
        "group_norm",
        input_types=(TensorType(DType.float32, shape, device_ref),),
    ) as graph:
        layer = GroupNorm(num_groups, num_channels, eps, affine)
        output = layer(graph.inputs[0].tensor)
        graph.output(output)
    if affine:
        assert weight is not None and bias is not None
        state_dict = {"weight": weight, "bias": bias}
        return gpu_session.load(graph, weights_registry=state_dict)
    return gpu_session.load(graph)


def torch_group_norm(
    input_tensor: torch.Tensor,
    num_groups: int,
    num_channels: int,
    eps: float,
    affine: bool = False,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    layer = torch.nn.GroupNorm(num_groups, num_channels, eps, affine)
    layer = layer.to(input_tensor.device)

    if affine:
        assert weight is not None and bias is not None
        layer.weight.data.copy_(weight.to(input_tensor.device))
        layer.bias.data.copy_(bias.to(input_tensor.device))

    return layer(input_tensor)


@pytest.mark.parametrize(
    "num_channels,num_groups,affine",
    [
        (2, 1, True),
        (4, 2, True),
        (6, 3, True),
        (2, 1, False),
        (4, 2, False),
    ],
)
def test_group_norm(
    gpu_session: InferenceSession,
    num_channels: int,
    num_groups: int,
    affine: bool,
) -> None:
    """Test group normalization with different configurations.

    Args:
        num_channels: Number of input channels
        num_groups: Number of groups to separate channels into
        affine: Whether to apply learnable affine transform
    """
    torch.manual_seed(42)

    # Create input tensor with shape (batch_size, channels, sequence_length)
    input_tensor = torch.randn((5, num_channels, 15), dtype=torch.float32)

    # Create weight and bias if using affine transform
    weight = torch.randn(num_channels) if affine else None
    bias = torch.randn(num_channels) if affine else None

    model = load_max_group_norm(
        gpu_session,
        input_tensor.shape,
        num_groups=num_groups,
        num_channels=num_channels,
        eps=1e-5,
        affine=affine,
        weight=weight,
        bias=bias,
    )

    input_tensor = input_tensor.to("cuda")
    torch_out = torch_group_norm(
        input_tensor,
        num_groups=num_groups,
        num_channels=num_channels,
        eps=1e-5,
        affine=affine,
        weight=weight,
        bias=bias,
    )
    max_out = torch.from_dlpack(model(input_tensor)[0])

    torch.testing.assert_close(
        torch_out,
        max_out,
        rtol=1e-6,
        atol=2 * torch.finfo(torch.float32).eps,
    )


def test_group_norm_invalid_input_shapes() -> None:
    with Graph(
        "group_norm",
        input_types=(TensorType(DType.float32, [10], DeviceRef.GPU()),),
    ) as graph:
        with pytest.raises(
            ValueError,
            match="Expected input tensor with >=2 dimensions, got shape",
        ):
            _ = GroupNorm(5, 10)(graph.inputs[0].tensor)

    with Graph(
        "group_norm",
        input_types=(TensorType(DType.float32, [5, 8], DeviceRef.GPU()),),
    ) as graph:
        with pytest.raises(ValueError, match="Expected 10 channels"):
            _ = GroupNorm(5, 10)(graph.inputs[0].tensor)


def test_group_norm_invalid_init() -> None:
    with pytest.raises(
        ValueError,
        match=r"num_channels.* should be divisible by num_groups.*",
    ):
        _ = GroupNorm(5, 11)
