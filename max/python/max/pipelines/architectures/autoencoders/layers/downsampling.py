# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Downsampling utilities for MAX framework."""

from max import functional as F
from max.dtype import DType
from max.graph import DeviceRef
from max.nn import Conv2d, Module
from max.nn.norm import LayerNorm, RMSNorm  # type: ignore[attr-defined]
from max.tensor import Tensor

class Downsample2D(Module[[Tensor], Tensor]):
    """A 2D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
        name (`str`, default `conv`):
            name of the downsampling 2D layer.
        kernel_size (`int`, default `3`):
            kernel size for the convolution.
        norm_type (`str`, optional):
            normalization type. Supported: "ln_norm" (LayerNorm), "rms_norm" (RMSNorm), or None.
        eps (`float`, optional):
            epsilon for normalization. Defaults to 1e-5 for LayerNorm, 1e-6 for RMSNorm.
        elementwise_affine (`bool`, optional):
            elementwise affine for normalization. Only used for LayerNorm. Defaults to True.
        bias (`bool`, default `True`):
            whether to use bias in the convolution.
    """

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
        """Initialize 2D downsampling module.

        Args:
            channels: Number of input channels.
            use_conv: Whether to use convolution for downsampling.
            out_channels: Number of output channels. If None, uses channels.
            padding: Padding for the convolution.
            name: Name for the convolution layer (unused, kept for compatibility).
            kernel_size: Kernel size for the convolution.
            norm_type: Normalization type ("ln_norm", "rms_norm", or None).
            eps: Epsilon for normalization. Defaults to 1e-5 for LayerNorm, 1e-6 for RMSNorm.
            elementwise_affine: Elementwise affine for normalization. Only used for LayerNorm.
            bias: Whether to use bias in the convolution.
            device: Device reference for module placement.
            dtype: Data type for module parameters.
        """
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        # Initialize normalization layer if specified
        if norm_type == "ln_norm":
            self.norm = LayerNorm(
                dim=channels,
                eps=eps or 1e-5,
                elementwise_affine=elementwise_affine if elementwise_affine is not None else True,
            )
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(dim=channels, eps=eps or 1e-6)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        if use_conv:
            self.conv = Conv2d(
                kernel_size=kernel_size,
                in_channels=self.channels,
                out_channels=self.out_channels,
                dtype=dtype,
                stride=stride,
                padding=padding,
                has_bias=bias,
                device=device,
                permute=True,
            )
        else:
            # Use avg_pool2d when use_conv=False (called in forward)
            if self.channels != self.out_channels:
                raise ValueError(
                    f"When use_conv=False, channels must equal out_channels. "
                    f"Got channels={self.channels}, out_channels={self.out_channels}"
                )
            # avg_pool2d is a function, not a module, so we call it in forward
            self.conv = None

    def forward(self, hidden_states: Tensor, *args, **kwargs) -> Tensor:
        """Apply 2D downsampling with optional convolution.

        Args:
            hidden_states: Input tensor of shape [N, C, H, W].
            *args: Additional positional arguments (ignored, kept for compatibility).
            **kwargs: Additional keyword arguments (ignored, kept for compatibility).

        Returns:
            Downsampled tensor of shape [N, C_out, H//2, W//2].
        """
        # Apply normalization if specified
        # Note: LayerNorm and RMSNorm normalize over the last dimension
        # Diffusers does: norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # This permutes [N, C, H, W] -> [N, H, W, C] -> norm -> [N, C, H, W]
        if self.norm is not None:
            # Permute to [N, H, W, C] for normalization
            hidden_states = F.permute(hidden_states, [0, 2, 3, 1])
            hidden_states = self.norm(hidden_states)
            # Permute back to [N, C, H, W]
            hidden_states = F.permute(hidden_states, [0, 3, 1, 2])

        # Handle padding=0 case: add padding before convolution
        # Diffusers: F.pad(hidden_states, (0, 1, 0, 1), mode="constant", value=0)
        # PyTorch pad format for 4D [N, C, H, W]: (pad_left, pad_right, pad_top, pad_bottom)
        # Max pad format for 4D [N, C, H, W]: [pad_before_N, pad_after_N, pad_before_C, pad_after_C,
        #                                      pad_before_H, pad_after_H, pad_before_W, pad_after_W]
        if self.use_conv and self.padding == 0:
            # PyTorch (0, 1, 0, 1) means: pad_right=1 on W, pad_bottom=1 on H
            # Max format: [0, 0, 0, 0, 0, 1, 0, 1]
            # (no padding on N and C, pad_after_H=1, pad_after_W=1)
            paddings = [0, 0, 0, 0, 0, 1, 0, 1]
            hidden_states = F.pad(hidden_states, paddings=paddings, mode="constant", value=0)

        # Apply downsampling
        if self.use_conv:
            hidden_states = self.conv(hidden_states)
        else:
            # Use avg_pool2d for downsampling
            # avg_pool2d expects [N, H, W, C] format
            # Permute to [N, H, W, C]
            hidden_states = F.permute(hidden_states, [0, 2, 3, 1])
            # Apply avg_pool2d with kernel_size=2, stride=2
            hidden_states = F.avg_pool2d(
                hidden_states,
                kernel_size=(2, 2),
                stride=2,
                padding=0,
            )
            hidden_states = F.permute(hidden_states, [0, 3, 1, 2])

        return hidden_states
