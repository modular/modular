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

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import DeviceRef
from max.nn.module_v3 import Conv2d, GroupNorm, Module


class ResnetBlock2D(Module[[Tensor, Tensor | None], Tensor]):
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = use_conv_shortcut

        self.norm1 = GroupNorm(
            num_groups=groups,
            num_channels=in_channels,
            eps=eps,
            affine=True,
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
        )

        self.norm2 = GroupNorm(
            num_groups=groups_out,
            num_channels=out_channels,
            eps=eps,
            affine=True,
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
        )

        self.conv_shortcut: Conv2d | None = None
        if self.use_conv_shortcut:
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
            )
        elif in_channels != out_channels:
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
            )

    def forward(self, x: Tensor, temb: Tensor | None = None) -> Tensor:
        """Apply ResnetBlock2D forward pass.

        Args:
            x: Input tensor of shape [N, C, H, W].
            temb: Optional time embedding tensor (currently unused).

        Returns:
            Output tensor of shape [N, C_out, H, W] with residual connection.
        """
        shortcut = (
            self.conv_shortcut(x) if self.conv_shortcut is not None else x
        )

        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        h = F.silu(self.norm2(h))
        h = self.conv2(h)

        return h + shortcut
