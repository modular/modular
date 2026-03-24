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
from max.nn.conv import Conv2d
from max.nn.layer import Module


def interpolate_2d_nearest(
    x: TensorValue,
    scale_factor: int = 2,
) -> TensorValue:
    """Nearest-neighbor upsampling using reshape+broadcast."""

    if x.rank != 4:
        raise ValueError(f"Input tensor must have rank 4, got {x.rank}")
    if scale_factor != 2:
        raise NotImplementedError(
            f"Only scale_factor=2 is currently supported, got {scale_factor}"
        )

    batch, channels, height, width = x.shape
    x = ops.reshape(x, [batch, channels, height, 1, width, 1])
    ones = ops.broadcast_to(
        ops.constant(1.0, dtype=x.dtype, device=x.device),
        [1, 1, 1, scale_factor, 1, scale_factor],
    )
    return ops.reshape(
        x * ones,
        [batch, channels, height * scale_factor, width * scale_factor],
    )


class Upsample2D(Module):
    """2D upsampling module for the V2 VAE."""

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: int | None = None,
        name: str = "conv",
        kernel_size: int | None = None,
        padding: int = 1,
        bias: bool = True,
        interpolate: bool = True,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__()
        del name
        if dtype is None:
            raise ValueError("dtype must be set for Upsample2D")
        if device is None:
            raise ValueError("device must be set for Upsample2D")
        if use_conv_transpose:
            raise NotImplementedError(
                "Upsample2D does not support use_conv_transpose=True yet."
            )

        self.interpolate = interpolate
        self.use_conv = use_conv
        self.conv: Conv2d | None = None
        if self.use_conv:
            self.conv = Conv2d(
                kernel_size=3 if kernel_size is None else kernel_size,
                in_channels=channels,
                out_channels=out_channels or channels,
                dtype=dtype,
                stride=1,
                padding=padding,
                has_bias=bias,
                device=device,
                permute=True,
            )

    def __call__(self, x: TensorValue) -> TensorValue:
        if self.interpolate:
            x = interpolate_2d_nearest(x, scale_factor=2)
        if self.use_conv and self.conv is not None:
            x = self.conv(x)
        return x
