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
from max.graph import DeviceRef, TensorValue, ops
from max.graph.ops import avg_pool2d
from max.nn.conv import Conv2d
from max.nn.layer import Module
from max.nn.norm import LayerNorm, RMSNorm


class Downsample2D(Module):
    """2D downsampling layer for the V2 VAE."""

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: int | None = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size: int = 3,
        norm_type: str | None = None,
        eps: float | None = None,
        elementwise_affine: bool | None = None,
        bias: bool = True,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__()
        del name
        if dtype is None:
            raise ValueError("dtype must be set for Downsample2D")
        if device is None:
            raise ValueError("device must be set for Downsample2D")

        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding

        self.norm: LayerNorm | RMSNorm | None = None
        if norm_type == "ln_norm":
            self.norm = LayerNorm(
                dims=channels,
                devices=[device],
                dtype=dtype,
                eps=eps or 1e-5,
                use_bias=(
                    True if elementwise_affine is None else elementwise_affine
                ),
            )
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(
                dim=channels,
                dtype=dtype,
                eps=eps or 1e-6,
            )
        elif norm_type is not None:
            raise ValueError(f"unknown norm_type: {norm_type}")

        self.conv: Conv2d | None = None
        if use_conv:
            self.conv = Conv2d(
                kernel_size=kernel_size,
                in_channels=channels,
                out_channels=self.out_channels,
                dtype=dtype,
                stride=2,
                padding=padding,
                has_bias=bias,
                device=device,
                permute=True,
            )
        elif channels != self.out_channels:
            raise ValueError(
                "When use_conv=False, channels must equal out_channels."
            )

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        if self.norm is not None:
            hidden_states = ops.permute(hidden_states, [0, 2, 3, 1])
            hidden_states = self.norm(hidden_states)
            hidden_states = ops.permute(hidden_states, [0, 3, 1, 2])

        if self.use_conv and self.padding == 0:
            hidden_states = ops.pad(hidden_states, [0, 0, 0, 0, 0, 1, 0, 1])

        if self.use_conv:
            assert self.conv is not None
            return self.conv(hidden_states)

        hidden_states = ops.permute(hidden_states, [0, 2, 3, 1])
        hidden_states = avg_pool2d(
            hidden_states,
            kernel_size=(2, 2),
            stride=2,
            padding=0,
        )
        return ops.permute(hidden_states, [0, 3, 1, 2])
