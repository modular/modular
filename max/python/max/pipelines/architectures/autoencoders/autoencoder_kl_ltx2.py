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
"""LTX2 Video Autoencoder Architecture."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import max.experimental.functional as F
from max.driver import Device
from max.dtype import DType
from max.experimental import nn, random
from max.experimental.nn.common_layers.activation import (
    activation_function_from_name,
)
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding

from .embeddings import PixArtAlphaCombinedTimestepSizeEmbeddings
from .model import BaseAutoencoderModel
from .model_config import (
    AutoencoderKLLTX2VideoConfig,
)


def pixel_shuffle_3d_merge(x: Tensor, stride: tuple[int, int, int]) -> Tensor:
    """Robust 3D pixel shuffle merge.

    Input x is rank 8: [B, C, D, d, H, h, W, w]
    Output is rank 5: [B, C, D*d, H*h, W*w]

    Uses a single direct reshape with explicit symbolic products (no -1 inference), because the stepwise flatten approach fails when
    the symbolic verifier encounters pre-existing mul_no_wrap products in later
    merge steps.  All input dims must be pure (non-product) Dims.
    """
    b, c, D, d, H, h, W, w = x.shape
    return x.reshape((b, c, D * d, H * h, W * w))


class PerChannelRMSNorm(nn.Module[[Tensor], Tensor]):
    """Per-channel RMS normalization layer."""

    def __init__(self, channel_dim: int = 1, eps: float = 1e-8) -> None:
        super().__init__()
        self.channel_dim = channel_dim
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean_sq = F.mean(x**2, axis=self.channel_dim)
        rms = F.sqrt(mean_sq + self.eps)
        return x / rms


class LTX2VideoCausalConv3d(nn.Module[..., Tensor]):
    """Causal or non-causal 3D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        spatial_padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size, kernel_size)
        )

        dilation = dilation if isinstance(dilation, tuple) else (dilation, 1, 1)
        stride = (
            stride if isinstance(stride, tuple) else (stride, stride, stride)
        )

        # Spatial padding (height and width)
        self.height_pad = self.kernel_size[1] // 2
        self.width_pad = self.kernel_size[2] // 2
        self.spatial_padding_mode = spatial_padding_mode

        if spatial_padding_mode == "zeros":
            # Let Conv3d handle zero-padding directly
            padding = (0, self.height_pad, self.width_pad)
        else:
            # We'll apply reflect (or other) padding manually before conv
            padding = (0, 0, 0)

        self.conv = nn.Conv3d(
            kernel_size=self.kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            dilation=dilation,
            num_groups=groups,
            padding=padding,
            permute=True,
        )

    def _reflect_pad_axis(self, x: Tensor, pad: int, axis: int) -> Tensor:
        """Apply reflect padding along a single axis of x.

        For a 5D tensor [B, C, D, H, W]:
          pad=1 on axis=3 (H):
            left  = x[:, :, :, 1:2, :]       (index 1)
            right = x[:, :, :, -2:-1, :]     (index n-2)
          pad=2 on axis=3 (H):
            left  = x[:, :, :, 2:3, :], x[:, :, :, 1:2, :]   (indices 2, 1)
            right = x[:, :, :, -2:-1, :], x[:, :, :, -3:-2, :] (indices n-2, n-3)
        """
        if pad == 0:
            return x

        # Build reflected slices for the left side (indices pad, pad-1, ..., 1)
        left_slices = []
        for i in range(pad, 0, -1):
            # Slice a single element at index i along `axis`
            slc = [slice(None)] * 5
            slc[axis] = slice(i, i + 1)
            left_slices.append(x[tuple(slc)])

        # Build reflected slices for the right side (indices n-2, n-3, ..., n-1-pad)
        right_slices = []
        for i in range(2, pad + 2):
            slc = [slice(None)] * 5
            slc[axis] = slice(-i, -i + 1 if -i + 1 != 0 else None)
            right_slices.append(x[tuple(slc)])

        return F.concat([*left_slices, x, *right_slices], axis=axis)

    def forward(self, hidden_states: Tensor, causal: bool = True) -> Tensor:
        tk = self.kernel_size[0]

        if causal:
            # Pad left (past) for causality
            # x shape: [B, C, D, H, W]
            pad_left = F.concat(
                [hidden_states[:, :, :1, :, :]] * (tk - 1), axis=2
            )
            hidden_states = F.concat([pad_left, hidden_states], axis=2)
        else:
            # Pad left (past) and right (future) for non-causal
            pad_left = F.concat(
                [hidden_states[:, :, :1, :, :]] * (tk // 2), axis=2
            )
            pad_right = F.concat(
                [hidden_states[:, :, -1:, :, :]] * (tk // 2), axis=2
            )
            hidden_states = F.concat(
                [pad_left, hidden_states, pad_right], axis=2
            )

        # Apply spatial padding manually when mode is not "zeros"
        # (Conv3d only supports zero-padding; reflect must be done here)
        if self.spatial_padding_mode != "zeros":
            if self.height_pad > 0:
                hidden_states = self._reflect_pad_axis(
                    hidden_states, self.height_pad, axis=3
                )
            if self.width_pad > 0:
                hidden_states = self._reflect_pad_axis(
                    hidden_states, self.width_pad, axis=4
                )

        return self.conv(hidden_states)


class LTX2VideoResnetBlock3d(nn.Module[..., Tensor]):
    """3D ResNet block used in LTX2 Video decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        dropout: float = 0.0,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels

        self.nonlinearity = activation_function_from_name(non_linearity)

        self.norm1 = PerChannelRMSNorm()
        self.conv1 = LTX2VideoCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.norm2 = PerChannelRMSNorm()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = LTX2VideoCausalConv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.norm3: nn.LayerNorm | None = None
        self.conv_shortcut: nn.Module[[Tensor], Tensor] | None = None
        if in_channels != out_channels:
            self.norm3 = nn.LayerNorm(
                in_channels, eps=eps, elementwise_affine=True, use_bias=True
            )
            # LTX 2.0 uses a normal Conv3d here rather than LTXVideoCausalConv3d
            self.conv_shortcut = nn.Conv3d(
                kernel_size=1,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=1,
                permute=True,
            )

        self.per_channel_scale1: Tensor | None = None
        self.per_channel_scale2: Tensor | None = None
        if inject_noise:
            self.per_channel_scale1 = Tensor.constant(
                Tensor.zeros((in_channels, 1, 1))
            )
            self.per_channel_scale2 = Tensor.constant(
                Tensor.zeros((in_channels, 1, 1))
            )

        self.scale_shift_table: Tensor | None = None
        if timestep_conditioning:
            self.scale_shift_table = Tensor.constant(
                random.gaussian((4, in_channels)) / in_channels**0.5
            )

    def forward(
        self,
        inputs: Tensor,
        temb: Tensor | None = None,
        seed: int | None = None,
        causal: bool = True,
    ) -> Tensor:
        hidden_states = inputs

        hidden_states = self.norm1(hidden_states)

        if self.scale_shift_table is not None:
            assert temb is not None
            temb = (
                F.reshape(temb, (temb.shape[0], 4, -1))[..., None, None, None]
                + self.scale_shift_table[None, ..., None, None, None]
            )
            shift_1 = temb[:, 0]
            scale_1 = temb[:, 1]
            shift_2 = temb[:, 2]
            scale_2 = temb[:, 3]
            hidden_states = hidden_states * (1 + scale_1) + shift_1

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states, causal=causal)

        if self.per_channel_scale1 is not None:
            spatial_shape = hidden_states.shape[-2:]
            spatial_noise = random.gaussian(
                spatial_shape,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )[None]
            hidden_states = (
                hidden_states
                + (spatial_noise * self.per_channel_scale1)[None, :, None, ...]
            )

        hidden_states = self.norm2(hidden_states)

        if self.scale_shift_table is not None:
            hidden_states = hidden_states * (1 + scale_2) + shift_2

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states, causal=causal)

        if self.per_channel_scale2 is not None:
            spatial_shape = hidden_states.shape[-2:]
            spatial_noise = random.gaussian(
                spatial_shape,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )[None]
            hidden_states = (
                hidden_states
                + (spatial_noise * self.per_channel_scale2)[None, :, None, ...]
            )

        if self.norm3 is not None:
            inputs = self.norm3(inputs.permute([0, 4, 2, 3, 1])).permute(
                [0, 4, 2, 3, 1]
            )

        if self.conv_shortcut is not None:
            inputs = self.conv_shortcut(inputs)

        hidden_states = hidden_states + inputs
        return hidden_states


class LTXVideoDownsampler3d(nn.Module[..., Tensor]):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int | tuple[int, int, int] = 1,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()

        self.stride: tuple[int, int, int] = (
            stride if isinstance(stride, tuple) else (stride, stride, stride)
        )
        self.group_size = (
            in_channels * self.stride[0] * self.stride[1] * self.stride[2]
        ) // out_channels

        out_channels = out_channels // (
            self.stride[0] * self.stride[1] * self.stride[2]
        )

        self.conv = LTX2VideoCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: Tensor, causal: bool = True) -> Tensor:
        hidden_states = F.concat(
            [hidden_states[:, :, : self.stride[0] - 1], hidden_states], axis=2
        )

        N, C, D, H, W = hidden_states.shape

        # Rebind to satisfy symbolic shape matching
        depth, h_stride, w_stride = self.stride
        intermediate_5d_shape = (
            N,
            C,
            (D // depth) * depth,
            (H // h_stride) * h_stride,
            (W // w_stride) * w_stride,
        )
        residual = F.rebind(hidden_states, intermediate_5d_shape).reshape(
            (
                N,
                C,
                D // depth,
                depth,
                H // h_stride,
                h_stride,
                W // w_stride,
                w_stride,
            )
        )
        residual = residual.permute([0, 1, 3, 5, 7, 2, 4, 6])
        residual = F.flatten(residual, 1, 4)

        r_shape = residual.shape
        # Rebind to satisfy symbolic shape matching
        new_c = r_shape[1] // self.group_size
        intermediate_5d_shape = (
            r_shape[0],
            new_c * self.group_size,
            r_shape[2],
            r_shape[3],
            r_shape[4],
        )
        residual = F.rebind(residual, intermediate_5d_shape).reshape(
            (
                r_shape[0],
                new_c,
                self.group_size,
                r_shape[2],
                r_shape[3],
                r_shape[4],
            )
        )
        residual = residual.mean(axis=2)

        hidden_states = self.conv(hidden_states, causal=causal)

        N, C, D, H, W = hidden_states.shape
        intermediate_5d_shape = (
            N,
            C,
            (D // depth) * depth,
            (H // h_stride) * h_stride,
            (W // w_stride) * w_stride,
        )
        hidden_states = F.rebind(hidden_states, intermediate_5d_shape).reshape(
            (
                N,
                C,
                D // depth,
                depth,
                H // h_stride,
                h_stride,
                W // w_stride,
                w_stride,
            )
        )
        hidden_states = hidden_states.permute([0, 1, 3, 5, 7, 2, 4, 6])
        hidden_states = F.flatten(hidden_states, 1, 4)

        hidden_states = hidden_states + residual

        return hidden_states


class LTXVideoUpsampler3d(nn.Module[..., Tensor]):
    def __init__(
        self,
        in_channels: int,
        stride: int | tuple[int, int, int] = 1,
        residual: bool = False,
        upscale_factor: int = 1,
        spatial_padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        self.stride: tuple[int, int, int] = (
            stride if isinstance(stride, tuple) else (stride, stride, stride)
        )
        self.residual = residual
        self.upscale_factor = upscale_factor

        out_channels = (
            in_channels * self.stride[0] * self.stride[1] * self.stride[2]
        ) // upscale_factor

        self.conv = LTX2VideoCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: Tensor, causal: bool = True) -> Tensor:
        batch_size, num_channels, num_frames, height, width = (
            hidden_states.shape
        )
        stride_prod = self.stride[0] * self.stride[1] * self.stride[2]

        if self.residual:
            residual = hidden_states.reshape(
                (
                    batch_size,
                    num_channels // stride_prod,
                    self.stride[0],
                    self.stride[1],
                    self.stride[2],
                    num_frames,
                    height,
                    width,
                )
            )
            residual = residual.permute([0, 1, 5, 2, 6, 3, 7, 4])
            depth, h_stride, w_stride = self.stride
            # Use concat-based merge to avoid symbolic product issues in reshape
            residual = pixel_shuffle_3d_merge(
                residual, (depth, h_stride, w_stride)
            )
            # Rebind to clean symbolic shape for downstream ops
            residual = F.rebind(
                residual,
                (
                    batch_size,
                    num_channels // stride_prod,
                    num_frames * depth,
                    height * h_stride,
                    width * w_stride,
                ),
            )
            # Already 5D [B, C, D, H, W]
            repeats = (
                self.stride[0] * self.stride[1] * self.stride[2]
            ) // self.upscale_factor
            if repeats > 1:
                # Use concat instead of tile for 5D
                residual = F.concat([residual] * repeats, axis=1)
            residual = residual[:, :, self.stride[0] - 1 :]

        hidden_states = self.conv(hidden_states, causal=causal)
        num_channels = hidden_states.shape[1]
        hidden_states = hidden_states.reshape(
            (
                batch_size,
                num_channels // stride_prod,
                self.stride[0],
                self.stride[1],
                self.stride[2],
                num_frames,
                height,
                width,
            )
        )
        hidden_states = hidden_states.permute([0, 1, 5, 2, 6, 3, 7, 4])
        depth, h_stride, w_stride = self.stride
        # Use concat-based merge to avoid symbolic product issues in reshape
        hidden_states = pixel_shuffle_3d_merge(
            hidden_states, (depth, h_stride, w_stride)
        )
        # Rebind to clean symbolic shape for downstream ops
        hidden_states = F.rebind(
            hidden_states,
            (
                batch_size,
                num_channels // stride_prod,
                num_frames * depth,
                height * h_stride,
                width * w_stride,
            ),
        )
        # Already 5D [B, C, D, H, W]
        hidden_states = hidden_states[:, :, self.stride[0] - 1 :]

        if self.residual:
            hidden_states = hidden_states + residual

        return hidden_states


class LTX2VideoDownBlock3D(nn.Module[..., Tensor]):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        spatio_temporal_scale: bool = True,
        downsample_type: str = "conv",
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                LTX2VideoResnetBlock3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    spatial_padding_mode=spatial_padding_mode,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.downsamplers = None
        if spatio_temporal_scale:
            self.downsamplers = nn.ModuleList()

            if downsample_type == "conv":
                self.downsamplers.append(
                    LTX2VideoCausalConv3d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=3,
                        stride=(2, 2, 2),
                        spatial_padding_mode=spatial_padding_mode,
                    )
                )
            elif downsample_type == "spatial":
                self.downsamplers.append(
                    LTXVideoDownsampler3d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=(1, 2, 2),
                        spatial_padding_mode=spatial_padding_mode,
                    )
                )
            elif downsample_type == "temporal":
                self.downsamplers.append(
                    LTXVideoDownsampler3d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=(2, 1, 1),
                        spatial_padding_mode=spatial_padding_mode,
                    )
                )
            elif downsample_type == "spatiotemporal":
                self.downsamplers.append(
                    LTXVideoDownsampler3d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=(2, 2, 2),
                        spatial_padding_mode=spatial_padding_mode,
                    )
                )

    def forward(
        self,
        hidden_states: Tensor,
        temb: Tensor | None = None,
        seed: int | None = None,
        causal: bool = True,
    ) -> Tensor:
        r"""Forward method of the `LTXDownBlock3D` class."""

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb, seed, causal=causal)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, causal=causal)

        return hidden_states


class LTX2VideoMidBlock3d(nn.Module[..., Tensor]):
    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        self.time_embedder = None
        if timestep_conditioning:
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(
                in_channels * 4, 0
            )

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                LTX2VideoResnetBlock3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    inject_noise=inject_noise,
                    timestep_conditioning=timestep_conditioning,
                    spatial_padding_mode=spatial_padding_mode,
                )
            )
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self,
        hidden_states: Tensor,
        temb: Tensor | None = None,
        seed: int | None = None,
        causal: bool = True,
    ) -> Tensor:
        r"""Forward method of the `LTXMidBlock3D` class."""

        if self.time_embedder is not None:
            temb = self.time_embedder(
                timestep=F.flatten(temb),
                resolution=None,
                aspect_ratio=None,
                batch_size=hidden_states.shape[0],
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.reshape((hidden_states.shape[0], -1, 1, 1, 1))

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb, seed, causal=causal)

        return hidden_states


class LTX2VideoUpBlock3d(nn.Module[..., Tensor]):
    r"""
    Up block used in the LTXVideo model.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`, *optional*):
            Number of output channels. If None, defaults to `in_channels`.
        num_layers (`int`, defaults to `1`):
            Number of resnet layers.
        dropout (`float`, defaults to `0.0`):
            Dropout rate.
        resnet_eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        resnet_act_fn (`str`, defaults to `"swish"`):
            Activation function to use.
        spatio_temporal_scale (`bool`, defaults to `True`):
            Whether or not to use a downsampling layer. If not used, output dimension would be same as input dimension.
            Whether or not to downsample across temporal dimension.
        is_causal (`bool`, defaults to `True`):
            Whether this layer behaves causally (future frames depend only on past frames) or not.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        spatio_temporal_scale: bool = True,
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        upsample_residual: bool = False,
        upscale_factor: int = 1,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        self.time_embedder = None
        if timestep_conditioning:
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(
                in_channels * 4, 0
            )

        self.conv_in = None
        if in_channels != out_channels:
            self.conv_in = LTX2VideoResnetBlock3d(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout=dropout,
                eps=resnet_eps,
                non_linearity=resnet_act_fn,
                inject_noise=inject_noise,
                timestep_conditioning=timestep_conditioning,
                spatial_padding_mode=spatial_padding_mode,
            )

        self.upsamplers = None
        if spatio_temporal_scale:
            self.upsamplers = nn.ModuleList(
                [
                    LTXVideoUpsampler3d(
                        out_channels * upscale_factor,
                        stride=(2, 2, 2),
                        residual=upsample_residual,
                        upscale_factor=upscale_factor,
                        spatial_padding_mode=spatial_padding_mode,
                    )
                ]
            )

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                LTX2VideoResnetBlock3d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    inject_noise=inject_noise,
                    timestep_conditioning=timestep_conditioning,
                    spatial_padding_mode=spatial_padding_mode,
                )
            )
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self,
        hidden_states: Tensor,
        temb: Tensor | None = None,
        seed: int | None = None,
        causal: bool = True,
    ) -> Tensor:
        if self.conv_in is not None:
            hidden_states = self.conv_in(
                hidden_states, temb, seed, causal=causal
            )

        if self.time_embedder is not None:
            temb = self.time_embedder(
                timestep=F.flatten(temb),
                resolution=None,
                aspect_ratio=None,
                batch_size=hidden_states.shape[0],
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.reshape((hidden_states.shape[0], -1, 1, 1, 1))

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, causal=causal)

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb, seed, causal=causal)

        return hidden_states


class LTX2VideoDecoder3d(nn.Module[..., Tensor]):
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 3,
        block_out_channels: tuple[int, ...] = (256, 512, 1024),
        spatio_temporal_scaling: tuple[bool, ...] = (True, True, True),
        layers_per_block: tuple[int, ...] = (5, 5, 5, 5),
        patch_size: int = 4,
        patch_size_t: int = 1,
        resnet_norm_eps: float = 1e-6,
        is_causal: bool = False,
        inject_noise: tuple[bool, ...] = (False, False, False),
        timestep_conditioning: bool = False,
        upsample_residual: tuple[bool, ...] = (True, True, True),
        upsample_factor: tuple[int, ...] = (2, 2, 2),
        spatial_padding_mode: str = "reflect",
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels * patch_size_t * patch_size**2
        self.is_causal = is_causal
        self.device = device
        self.dtype = dtype
        self.in_channels = in_channels

        block_out_channels = tuple(reversed(block_out_channels))
        spatio_temporal_scaling = tuple(reversed(spatio_temporal_scaling))
        layers_per_block = tuple(reversed(layers_per_block))
        inject_noise = tuple(reversed(inject_noise))
        upsample_residual = tuple(reversed(upsample_residual))
        upsample_factor = tuple(reversed(upsample_factor))
        output_channel = block_out_channels[0]

        self.conv_in = LTX2VideoCausalConv3d(
            in_channels,
            output_channel,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.mid_block = LTX2VideoMidBlock3d(
            in_channels=output_channel,
            num_layers=layers_per_block[0],
            resnet_eps=resnet_norm_eps,
            inject_noise=inject_noise[0],
            timestep_conditioning=timestep_conditioning,
            spatial_padding_mode=spatial_padding_mode,
        )

        # up blocks
        num_block_out_channels = len(block_out_channels)
        self.up_blocks = nn.ModuleList([])
        for i in range(num_block_out_channels):
            input_channel = output_channel // upsample_factor[i]
            output_channel = block_out_channels[i] // upsample_factor[i]

            up_block = LTX2VideoUpBlock3d(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=layers_per_block[i + 1],
                resnet_eps=resnet_norm_eps,
                spatio_temporal_scale=spatio_temporal_scaling[i],
                inject_noise=inject_noise[i + 1],
                timestep_conditioning=timestep_conditioning,
                upsample_residual=upsample_residual[i],
                upscale_factor=upsample_factor[i],
                spatial_padding_mode=spatial_padding_mode,
            )

            self.up_blocks.append(up_block)

        # out
        self.norm_out = PerChannelRMSNorm()
        self.conv_act = activation_function_from_name("silu")
        self.conv_out = LTX2VideoCausalConv3d(
            output_channel,
            self.out_channels,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )

        # timestep embedding
        self.time_embedder = None
        self.scale_shift_table = None
        self.timestep_scale_multiplier = None
        if timestep_conditioning:
            self.timestep_scale_multiplier = Tensor.constant(
                1000.0, dtype=DType.float32
            )
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(
                output_channel * 2, 0
            )
            self.scale_shift_table = (
                random.gaussian((2, output_channel)) / output_channel**0.5
            )

    def forward(
        self,
        hidden_states: Tensor,
        temb: Tensor | None = None,
        causal: bool | None = None,
    ) -> Tensor:
        causal = causal or self.is_causal

        hidden_states = self.conv_in(hidden_states, causal=causal)

        if self.timestep_scale_multiplier is not None:
            assert temb is not None
            temb = temb * self.timestep_scale_multiplier

        hidden_states = self.mid_block(hidden_states, temb, causal=causal)

        for up_block in self.up_blocks:
            hidden_states = up_block(hidden_states, temb, causal=causal)

        hidden_states = self.norm_out(hidden_states)

        if self.time_embedder is not None:
            temb = self.time_embedder(
                timestep=F.flatten(temb),
                resolution=None,
                aspect_ratio=None,
                batch_size=hidden_states.shape[0],
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.reshape((hidden_states.shape[0], -1, 1, 1, 1)).reshape(
                (hidden_states.shape[0], 2, -1, 1, 1, 1)
            )
            assert temb is not None
            assert self.scale_shift_table is not None
            temb = temb + self.scale_shift_table[None, ..., None, None, None]
            shift = temb[:, 0]
            scale = temb[:, 1]
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states, causal=causal)

        p = self.patch_size
        p_t = self.patch_size_t

        batch_size, num_channels, num_frames, height, width = (
            hidden_states.shape
        )
        target_padding_channels = num_channels // (p_t * p * p)
        # Rebind to satisfy symbolic shape matching
        intermediate_5d_shape = (
            batch_size,
            target_padding_channels * p_t * p * p,
            num_frames,
            height,
            width,
        )
        hidden_states = F.rebind(hidden_states, intermediate_5d_shape).reshape(
            (
                batch_size,
                target_padding_channels,
                p_t,
                p,
                p,
                num_frames,
                height,
                width,
            ),
        )
        hidden_states = hidden_states.permute([0, 1, 5, 2, 6, 4, 7, 3])
        # Use concat-based merge to avoid symbolic product issues in reshape
        hidden_states = pixel_shuffle_3d_merge(hidden_states, (p_t, p, p))
        # Rebind to clean symbolic shape for downstream ops
        hidden_states = F.rebind(
            hidden_states,
            (
                batch_size,
                target_padding_channels,
                num_frames * p_t,
                height * p,
                width * p,
            ),
        )
        # Already 5D [B, C, D, H, W]

        return hidden_states

    def input_types(self) -> tuple[TensorType, ...]:
        return (
            TensorType(
                self.dtype,
                shape=[
                    "batch_size",
                    self.in_channels,
                    "num_frames",
                    "height",
                    "width",
                ],
                device=self.device,
            ),
        )


class AutoencoderKLLTX2Video(nn.Module[[Tensor, Tensor | None, bool], Tensor]):
    """Refactored LTX2 Video Autoencoder."""

    def __init__(self, config: AutoencoderKLLTX2VideoConfig) -> None:
        super().__init__()
        self.config = config
        self.decoder = LTX2VideoDecoder3d(
            in_channels=config.latent_channels,
            out_channels=config.out_channels,
            patch_size=config.patch_size,
            patch_size_t=config.patch_size_t,
            is_causal=config.decoder_causal,
            block_out_channels=config.decoder_block_out_channels,
            layers_per_block=config.decoder_layers_per_block,
            inject_noise=config.decoder_inject_noise,
            upsample_residual=config.upsample_residual,
            upsample_factor=config.upsample_factor,
            spatio_temporal_scaling=config.decoder_spatio_temporal_scaling,
            resnet_norm_eps=config.resnet_norm_eps,
            timestep_conditioning=config.timestep_conditioning,
            spatial_padding_mode=config.decoder_spatial_padding_mode,
            device=config.device,
            dtype=config.dtype,
        )

    def forward(
        self,
        z: Tensor,
        timestep: Tensor | None = None,
        causal: bool | None = None,
    ) -> Tensor:
        return self.decoder(z, timestep, causal=causal)


class PostprocessAndDecode(nn.Module[..., Tensor]):
    """Fused unpack + latents denorm + VAE decode + post-process in a single compiled graph.

    Eliminates the inter-graph boundaries and intermediate tensor materializations
    that previously existed between unpack, denormalization, vae.decode(), and
    video post-processing.

    Accepts packed latents in (B, S, C) shape where
    S = latent_num_frames * latent_height * latent_width and C = num_channels.
    The LTX-2 transformer uses patch_size = patch_size_t = 1 so the packing is
    trivially one token per latent voxel, just like Flux2.
    The temporal and spatial dimensions are conveyed via three 1-D shape-carrier
    tensors whose *lengths* (not values) encode latent_num_frames, latent_height,
    and latent_width as symbolic graph Dims, so a single compiled graph handles
    any video size without recompilation.
    """

    def __init__(
        self,
        decoder: LTX2VideoDecoder3d,
        latents_mean: Tensor,
        latents_std: Tensor,
        num_channels: int,
        device: DeviceRef,
        dtype: DType,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.latents_mean = latents_mean
        self.latents_std = latents_std
        self._num_channels = num_channels
        self._device = device
        self._dtype = dtype

    def forward(
        self,
        latents_bsc: Tensor,
        f_carrier: Tensor,
        h_carrier: Tensor,
        w_carrier: Tensor,
    ) -> Tensor:
        batch = latents_bsc.shape[0]
        C = latents_bsc.shape[2]
        # Extract latent dims from carrier shapes (symbolic Dims, not runtime values).
        f = f_carrier.shape[0]
        h = h_carrier.shape[0]
        w = w_carrier.shape[0]

        # Assert seq == f * h * w so the reshape verifier accepts it, then unpack
        # packed (B, S, C) -> (B, C, F, H, W).  LTX-2's transformer uses
        # patch_size = patch_size_t = 1 so each token is one latent voxel.
        latents_bsc = F.rebind(latents_bsc, [batch, f * h * w, C])
        latents_bfhwc = F.reshape(latents_bsc, (batch, f, h, w, C))
        latents = F.permute(latents_bfhwc, [0, 4, 1, 2, 3])  # (B, C, F, H, W)

        # Denormalize: latents * latents_std + latents_mean
        # (scaling_factor is always 1.0 for LTX-2 video VAE)
        latents_mean_r = F.reshape(self.latents_mean, (1, C, 1, 1, 1))
        latents_std_r = F.reshape(self.latents_std, (1, C, 1, 1, 1))
        latents = latents * latents_std_r + latents_mean_r

        decoded = self.decoder(latents, None, causal=False)

        # Post-process: [-1, 1] -> [0, 255] uint8, keeping the decoder's native
        # dtype to reduce bandwidth before the final cast.
        decoded = decoded * 0.5 + 0.5
        decoded = F.max(decoded, 0.0)
        decoded = F.min(decoded, 1.0)
        decoded = decoded * 255.0

        # Permute: (B, C, F, H, W) -> (B, F, H, W, C)
        decoded = F.permute(decoded, [0, 2, 3, 4, 1])
        return F.transfer_to(F.cast(decoded, DType.uint8), DeviceRef.CPU())

    def input_types(self) -> tuple[TensorType, ...]:
        return (
            TensorType(
                self._dtype,
                shape=["batch_size", "video_seq_len", self._num_channels],
                device=self._device,
            ),
            # Shape carriers: lengths encode latent dims as symbolic Dims.
            # Content is never read; only the shapes matter.
            TensorType(
                DType.float32,
                shape=["latent_num_frames"],
                device=DeviceRef.CPU(),
            ),
            TensorType(
                DType.float32, shape=["latent_height"], device=DeviceRef.CPU()
            ),
            TensorType(
                DType.float32, shape=["latent_width"], device=DeviceRef.CPU()
            ),
        )


class AutoencoderKLLTX2VideoModel(BaseAutoencoderModel):
    """ComponentModel wrapper for LTX2 Video Autoencoder."""

    latents_mean: Tensor
    latents_std: Tensor

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        super().__init__(
            config=config,
            encoding=encoding,
            devices=devices,
            weights=weights,
            config_class=AutoencoderKLLTX2VideoConfig,
            autoencoder_class=AutoencoderKLLTX2Video,
        )

    def load_model(self) -> Any:
        """Load and compile the decoder model with latent normalisation statistics.

        Extracts latents_mean and latents_std tensors from the weights (which
        diffusers stores as persistent buffers), then delegates to the base class
        for weight loading and model compilation.

        Returns:
            Compiled decoder model callable.

        Raises:
            ValueError: If latents_mean or latents_std are not found in the weights.
        """
        latents_stats: dict[str, Any] = {}

        for key, value in self.weights.items():
            if key in ("latents_mean", "latents_std"):
                weight_data = value.data()
                target_dtype = self.config.dtype
                if weight_data.dtype != target_dtype:
                    if weight_data.dtype.is_float() and target_dtype.is_float():
                        weight_data = weight_data.astype(target_dtype)
                latents_stats[key] = weight_data.data

        latents_mean_data = latents_stats.get("latents_mean")
        latents_std_data = latents_stats.get("latents_std")

        if latents_mean_data is None or latents_std_data is None:
            raise ValueError(
                "Latent normalisation statistics (latents_mean, latents_std) not "
                "found in weights. Make sure the checkpoint contains these buffers."
            )

        super().load_model()

        self.latents_mean = Tensor.from_dlpack(latents_mean_data).to(
            self.devices[0]
        )
        self.latents_std = Tensor.from_dlpack(latents_std_data).to(
            self.devices[0]
        )

        return self.model

    def build_fused_decode(
        self,
        device: Device,
        num_channels: int,
    ) -> Callable[..., Any]:
        """Build a fused unpack + postprocess + VAE decode compiled graph.

        Combines latent unpacking, denormalization, VAE decoding, and video
        post-processing into a single compiled graph.  Packed latents in
        (B, S, C) format are accepted directly (as produced by the denoiser).
        LTX-2's transformer uses patch_size = patch_size_t = 1 so the (B, S, C)
        token format maps one-to-one to latent voxels.  Three shape-carrier
        tensors convey the latent temporal and spatial dimensions as symbolic
        Dims so the same compiled graph handles any video resolution without
        recompilation.

        Args:
            device: Target device for the compiled graph.
            num_channels: Number of latent channels (self.latents_mean.shape[0]).

        Returns:
            Compiled callable taking (latents_bsc, f_carrier, h_carrier, w_carrier)
            and returning a (B, F, H, W, C) uint8 video tensor on CPU.
        """
        dtype = self.config.dtype
        device_ref = DeviceRef.from_device(device)

        fused_weights: dict[str, Any] = {}
        for key, value in self.weights.items():
            weight_data = value.data()
            if weight_data.dtype != dtype:
                if weight_data.dtype.is_float() and dtype.is_float():
                    weight_data = weight_data.astype(dtype)
            if key.startswith("decoder."):
                # decoder.X -> decoder.X (PostprocessAndDecode.decoder.X)
                fused_weights[key] = weight_data
            elif key.startswith("post_quant_conv."):
                # post_quant_conv.X -> decoder.post_quant_conv.X
                fused_weights[f"decoder.{key}"] = weight_data
            elif key == "latents_mean":
                fused_weights["latents_mean"] = weight_data
            elif key == "latents_std":
                fused_weights["latents_std"] = weight_data

        with F.lazy():
            autoencoder = AutoencoderKLLTX2Video(self.config)
            fused = PostprocessAndDecode(
                decoder=autoencoder.decoder,
                latents_mean=self.latents_mean,
                latents_std=self.latents_std,
                num_channels=num_channels,
                device=device_ref,
                dtype=dtype,
            )
            fused.to(device)
            self._fused_model = fused.compile(
                *fused.input_types(), weights=fused_weights
            )

        return self._fused_model

    @property
    def bn(self) -> SimpleNamespace:
        """Exposes latent normalisation statistics in a diffusers-compatible namespace.

        Returns a ``SimpleNamespace`` with ``latents_mean`` and ``latents_std``
        attributes, mirroring the ``bn`` property on :class:`AutoencoderKLFlux2Model`
        so that both can be consumed by shared pipeline code.

        Returns:
            SimpleNamespace: Object containing ``latents_mean`` and ``latents_std``.
        """
        return SimpleNamespace(
            latents_mean=self.latents_mean, latents_std=self.latents_std
        )
