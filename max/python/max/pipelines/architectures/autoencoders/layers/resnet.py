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
from max.graph import DeviceRef, TensorValue
from max.nn.activation import activation_function_from_name
from max.nn.conv import Conv2d
from max.nn.layer import Module
from max.nn.norm import GroupNorm


class ResnetBlock2D(Module):
    """Residual block for the graph-based V2 VAE stack."""

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
        super().__init__()
        del temb_channels
        if dtype is None:
            raise ValueError("dtype must be set for ResnetBlock2D")
        if device is None:
            raise ValueError("device must be set for ResnetBlock2D")

        self.activation = activation_function_from_name(non_linearity)
        self.norm1 = GroupNorm(
            num_groups=groups,
            num_channels=in_channels,
            eps=eps,
            affine=True,
            device=device,
        )
        self.conv1 = Conv2d(
            kernel_size=3,
            in_channels=in_channels,
            out_channels=out_channels,
            dtype=dtype,
            stride=1,
            padding=1,
            has_bias=True,
            device=device,
            permute=True,
        )
        self.norm2 = GroupNorm(
            num_groups=groups_out,
            num_channels=out_channels,
            eps=eps,
            affine=True,
            device=device,
        )
        self.conv2 = Conv2d(
            kernel_size=3,
            in_channels=out_channels,
            out_channels=out_channels,
            dtype=dtype,
            stride=1,
            padding=1,
            has_bias=True,
            device=device,
            permute=True,
        )
        self.conv_shortcut: Conv2d | None = None
        if use_conv_shortcut or in_channels != out_channels:
            self.conv_shortcut = Conv2d(
                kernel_size=1,
                in_channels=in_channels,
                out_channels=out_channels,
                dtype=dtype,
                stride=1,
                padding=0,
                has_bias=conv_shortcut_bias,
                device=device,
                permute=True,
            )

    def __call__(
        self, x: TensorValue, temb: TensorValue | None = None
    ) -> TensorValue:
        del temb
        shortcut = (
            self.conv_shortcut(x) if self.conv_shortcut is not None else x
        )
        hidden = self.activation(self.norm1(x))
        hidden = self.conv1(hidden)
        hidden = self.activation(self.norm2(hidden))
        hidden = self.conv2(hidden)
        return hidden + shortcut
