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

from typing import Literal

from max.dtype import DType
from max.graph import DeviceRef, TensorValue
from max.nn.activation import activation_function_from_name
from max.nn.conv import Conv2d
from max.nn.layer import Module
from max.nn.norm import GroupNorm


class ResnetBlock2D(Module):
    """Residual block for 2D VAE decoder.

    This module implements a residual block with two convolutional layers,
    group normalization, and optional shortcut connection. It supports
    time embedding conditioning and configurable activation functions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int | None,
        groups: int,
        groups_out: int,
        eps: float = 1e-6,
        non_linearity: str = "silu",
        use_conv_shortcut: bool = False,
        conv_shortcut_bias: bool = True,
        io_layout: Literal["nchw", "nhwc"] = "nchw",
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize ResnetBlock2D module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            temb_channels: Number of time embedding channels (None if not used).
            groups: Number of groups for first GroupNorm.
            groups_out: Number of groups for second GroupNorm.
            eps: Epsilon value for GroupNorm layers.
            non_linearity: Activation function name (e.g., "silu").
            use_conv_shortcut: Whether to use convolutional shortcut.
            conv_shortcut_bias: Whether to use bias in shortcut convolution.
            device: Device reference for module placement.
            dtype: Data type for module parameters.
        """
        super().__init__()
        del temb_channels
        if dtype is None:
            raise ValueError("dtype must be set for ResnetBlock2D")
        if device is None:
            raise ValueError("device must be set for ResnetBlock2D")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = use_conv_shortcut
        self.io_layout = io_layout
        self.activation = activation_function_from_name(non_linearity)
        self.norm1 = GroupNorm(
            num_groups=groups,
            num_channels=in_channels,
            eps=eps,
            affine=True,
            io_layout=io_layout,
            device=device,
        )
        self.conv1 = Conv2d(
            kernel_size=3,
            in_channels=in_channels,
            out_channels=out_channels,
            dtype=dtype,
            stride=1,
            padding=1,
            dilation=1,
            num_groups=1,
            has_bias=True,
            device=device,
            permute=True,
            io_layout=io_layout,
        )
        self.norm2 = GroupNorm(
            num_groups=groups_out,
            num_channels=out_channels,
            eps=eps,
            affine=True,
            io_layout=io_layout,
            device=device,
        )
        self.conv2 = Conv2d(
            kernel_size=3,
            in_channels=out_channels,
            out_channels=out_channels,
            dtype=dtype,
            stride=1,
            padding=1,
            dilation=1,
            num_groups=1,
            has_bias=True,
            device=device,
            permute=True,
            io_layout=io_layout,
        )
        self.conv_shortcut: Conv2d | None = None
        if self.use_conv_shortcut or in_channels != out_channels:
            self.conv_shortcut = Conv2d(
                kernel_size=1,
                in_channels=in_channels,
                out_channels=out_channels,
                dtype=dtype,
                stride=1,
                padding=0,
                dilation=1,
                num_groups=1,
                has_bias=conv_shortcut_bias,
                device=device,
                permute=True,
                io_layout=io_layout,
            )

    def __call__(
        self, x: TensorValue, temb: TensorValue | None = None
    ) -> TensorValue:
        """Apply ResnetBlock2D forward pass.

        Args:
            x: Input tensor of shape [N, C, H, W].
            temb: Optional time embedding tensor (currently unused).

        Returns:
            Output tensor of shape [N, C_out, H, W] with residual connection.
        """
        del temb
        shortcut = (
            self.conv_shortcut(x) if self.conv_shortcut is not None else x
        )
        h = self.activation(self.norm1(x))
        h = self.conv1(h)
        h = self.activation(self.norm2(h))
        h = self.conv2(h)
        return h + shortcut
