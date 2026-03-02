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
"""LTX2 Audio Autoencoder Architecture."""

from typing import Any

import max.experimental.functional as F
import max.nn.module_v3 as nn
from max.driver import Device
from max.dtype import DType
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding

from .layers.upsampling import interpolate_2d_nearest
from .model import BaseAutoencoderModel
from .model_config import AutoencoderKLLTX2AudioConfig

LATENT_DOWNSAMPLE_FACTOR = 4


class LTX2AudioCausalConv2d(nn.Module[[Tensor], Tensor]):
    """A causal 2D convolution that pads asymmetrically along the causal axis."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        causality_axis: str = "height",
    ) -> None:
        super().__init__()
        self.causality_axis = causality_axis
        kernel_size = (
            (kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else kernel_size
        )
        dilation = (
            (dilation, dilation) if isinstance(dilation, int) else dilation
        )

        pad_h = (kernel_size[0] - 1) * dilation[0]
        pad_w = (kernel_size[1] - 1) * dilation[1]

        if self.causality_axis == "none":
            self.padding = (
                0,
                0,
                0,
                0,
                pad_h // 2,
                pad_h - pad_h // 2,
                pad_w // 2,
                pad_w - pad_w // 2,
            )
        elif self.causality_axis in {"width", "width-compatibility"}:
            self.padding = (
                0,
                0,
                0,
                0,
                pad_h // 2,
                pad_h - pad_h // 2,
                pad_w,
                0,
            )
        elif self.causality_axis == "height":
            self.padding = (
                0,
                0,
                0,
                0,
                pad_h,
                0,
                pad_w // 2,
                pad_w - pad_w // 2,
            )
        else:
            raise ValueError(f"Invalid causality_axis: {causality_axis}")

        self.conv = nn.Conv2d(
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            padding=0,
            dilation=dilation,
            num_groups=groups,
            has_bias=bias,
            permute=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, self.padding)
        return self.conv(x)


class LTX2AudioPixelNorm(nn.Module[[Tensor], Tensor]):
    """Per-pixel (per-location) RMS normalization layer."""

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean_sq = F.mean(x**2, axis=self.dim)
        rms = F.sqrt(mean_sq + self.eps)
        return x / rms


class LTX2AudioAttnBlock(nn.Module[[Tensor], Tensor]):
    """Attention block for LTX2 Audio."""

    def __init__(
        self,
        in_channels: int,
        norm_type: str = "group",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        if norm_type == "group":
            self.norm = nn.GroupNorm(
                num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
            )
        elif norm_type == "pixel":
            self.norm = LTX2AudioPixelNorm(dim=1, eps=1e-6)
        else:
            raise ValueError(f"Invalid normalization type: {norm_type}")

        self.q = nn.Conv2d(
            kernel_size=1,
            in_channels=in_channels,
            out_channels=in_channels,
            permute=True,
        )
        self.k = nn.Conv2d(
            kernel_size=1,
            in_channels=in_channels,
            out_channels=in_channels,
            permute=True,
        )
        self.v = nn.Conv2d(
            kernel_size=1,
            in_channels=in_channels,
            out_channels=in_channels,
            permute=True,
        )
        self.proj_out = nn.Conv2d(
            kernel_size=1,
            in_channels=in_channels,
            out_channels=in_channels,
            permute=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        batch, channels, height, width = q.shape
        # Use F.bmm for attention computation
        q = q.reshape((batch, channels, height * width)).permute((0, 2, 1))
        k = k.reshape((batch, channels, height * width))

        attn = F.bmm(q, k) * channels ** (-0.5)
        attn = F.softmax(attn, axis=-1)

        v = v.reshape((batch, channels, height * width))
        attn = attn.permute((0, 2, 1))

        h = F.bmm(v, attn).reshape((batch, channels, height, width))

        return x + self.proj_out(h)


class LTX2AudioResnetBlock(nn.Module[[Tensor, Tensor | None], Tensor]):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        norm_type: str = "group",
        causality_axis: str = "height",
    ) -> None:
        super().__init__()
        self.causality_axis = causality_axis

        if (
            self.causality_axis is not None
            and self.causality_axis != "none"
            and norm_type == "group"
        ):
            raise ValueError(
                "Causal ResnetBlock with GroupNorm is not supported."
            )
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        if norm_type == "group":
            self.norm1 = nn.GroupNorm(
                num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
            )
        elif norm_type == "pixel":
            self.norm1 = LTX2AudioPixelNorm(dim=1, eps=1e-6)
        else:
            raise ValueError(f"Invalid normalization type: {norm_type}")
        self.non_linearity = nn.SiLU()
        if causality_axis is not None:
            self.conv1 = LTX2AudioCausalConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                causality_axis=causality_axis,
            )
        else:
            self.conv1 = nn.Conv2d(
                kernel_size=3,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=1,
                padding=1,
                permute=True,
            )
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        if norm_type == "group":
            self.norm2 = nn.GroupNorm(
                num_groups=32, num_channels=out_channels, eps=1e-6, affine=True
            )
        elif norm_type == "pixel":
            self.norm2 = LTX2AudioPixelNorm(dim=1, eps=1e-6)
        else:
            raise ValueError(f"Invalid normalization type: {norm_type}")
        self.dropout = nn.Dropout(dropout)
        if causality_axis is not None:
            self.conv2 = LTX2AudioCausalConv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                causality_axis=causality_axis,
            )
        else:
            self.conv2 = nn.Conv2d(
                kernel_size=3,
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                padding=1,
                permute=True,
            )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                if causality_axis is not None:
                    self.conv_shortcut = LTX2AudioCausalConv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        causality_axis=causality_axis,
                    )
                else:
                    self.conv_shortcut = nn.Conv2d(
                        kernel_size=3,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=1,
                        padding=1,
                        permute=True,
                    )
            else:
                if causality_axis is not None:
                    self.nin_shortcut = LTX2AudioCausalConv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        causality_axis=causality_axis,
                    )
                else:
                    self.nin_shortcut = nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        permute=True,
                    )

    def forward(self, x: Tensor, temb: Tensor | None = None) -> Tensor:
        h = self.norm1(x)
        h = self.non_linearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(self.non_linearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.non_linearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = (
                self.conv_shortcut(x)
                if self.use_conv_shortcut
                else self.nin_shortcut(x)
            )

        return x + h


class LTX2AudioUpsample(nn.Module[[Tensor], Tensor]):
    def __init__(
        self,
        in_channels: int,
        with_conv: bool,
        causality_axis: str | None = "height",
    ) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis
        if self.with_conv:
            if causality_axis is not None:
                self.conv = LTX2AudioCausalConv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    causality_axis=causality_axis,
                )
            else:
                self.conv = nn.Conv2d(
                    kernel_size=3,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    stride=1,
                    padding=1,
                    permute=True,
                )

    def forward(self, x: Tensor) -> Tensor:
        x = interpolate_2d_nearest(x, scale_factor=2)
        if self.with_conv:
            x = self.conv(x)
            if self.causality_axis is None or self.causality_axis == "none":
                pass
            elif self.causality_axis == "height":
                x = x[:, :, 1:, :]
            elif self.causality_axis == "width":
                x = x[:, :, :, 1:]
            elif self.causality_axis == "width-compatibility":
                pass
            else:
                raise ValueError(
                    f"Invalid causality_axis: {self.causality_axis}"
                )

        return x


class LTX2AudioAudioPatchifier(nn.Module[[Tensor], Tensor]):
    """Patchifier for spectrogram/audio latents."""

    def __init__(
        self,
        patch_size: int,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
        is_causal: bool = True,
    ):
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.audio_latent_downsample_factor = audio_latent_downsample_factor
        self.is_causal = is_causal
        self._patch_size = (1, patch_size, patch_size)

    def patchify(self, audio_latents: Tensor) -> Tensor:
        batch, channels, time, freq = audio_latents.shape
        return audio_latents.permute((0, 2, 1, 3)).reshape(
            (batch, time, channels * freq)
        )

    def unpatchify(
        self, audio_latents: Tensor, channels: int, mel_bins: int
    ) -> Tensor:
        batch, time, _ = audio_latents.shape
        return audio_latents.reshape((batch, time, channels, mel_bins)).permute(
            (0, 2, 1, 3)
        )

    @property
    def patch_size(self) -> tuple[int, int, int]:
        return self._patch_size


class LTX2AudioDecoder(nn.Module[[Tensor], Tensor]):
    """
    Symmetric decoder that reconstructs audio spectrograms from latent features.

    The decoder mirrors the encoder structure with configurable channel multipliers, attention resolutions, and causal
    convolutions.
    """

    def __init__(
        self,
        base_channels: int = 128,
        output_channels: int = 1,
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] | None = None,
        in_channels: int = 2,
        resolution: int = 256,
        latent_channels: int = 8,
        ch_mult: tuple[int, ...] = (1, 2, 4),
        norm_type: str = "group",
        causality_axis: str | None = "width",
        dropout: float = 0.0,
        mid_block_add_attention: bool = False,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        is_causal: bool = True,
        mel_bins: int | None = 64,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__()

        self.sample_rate = sample_rate
        self.mel_hop_length = mel_hop_length
        self.is_causal = is_causal
        self.mel_bins = mel_bins
        self.device = device
        self.dtype = dtype
        self.patchifier = LTX2AudioAudioPatchifier(
            patch_size=1,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            sample_rate=sample_rate,
            hop_length=mel_hop_length,
            is_causal=is_causal,
        )

        self.base_channels = base_channels
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_ch = output_channels
        self.give_pre_end = False
        self.tanh_out = False
        self.norm_type = norm_type
        self.latent_channels = latent_channels
        self.channel_multipliers = ch_mult
        self.attn_resolutions = attn_resolutions
        self.causality_axis = causality_axis

        base_block_channels = base_channels * self.channel_multipliers[-1]
        base_resolution = resolution // (2 ** (self.num_resolutions - 1))
        self.z_shape = (1, latent_channels, base_resolution, base_resolution)

        if self.causality_axis is not None:
            self.conv_in = LTX2AudioCausalConv2d(
                latent_channels,
                base_block_channels,
                kernel_size=3,
                stride=1,
                causality_axis=self.causality_axis,
            )
        else:
            self.conv_in = nn.Conv2d(
                kernel_size=3,
                in_channels=latent_channels,
                out_channels=base_block_channels,
                stride=1,
                padding=1,
                permute=True,
            )
        self.non_linearity = nn.SiLU()
        self.mid = nn.Module()
        self.mid.block_1 = LTX2AudioResnetBlock(
            in_channels=base_block_channels,
            out_channels=base_block_channels,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
        )
        if mid_block_add_attention:
            self.mid.attn_1 = LTX2AudioAttnBlock(
                base_block_channels, norm_type=self.norm_type
            )
        else:
            self.mid.attn_1 = nn.Identity()
        self.mid.block_2 = LTX2AudioResnetBlock(
            in_channels=base_block_channels,
            out_channels=base_block_channels,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
        )

        self.up = nn.ModuleList()
        block_in = base_block_channels
        curr_res = self.resolution // (2 ** (self.num_resolutions - 1))

        for level in reversed(range(self.num_resolutions)):
            stage = nn.Module()
            stage.block = nn.ModuleList()
            stage.attn = nn.ModuleList()
            block_out = self.base_channels * self.channel_multipliers[level]

            for _ in range(self.num_res_blocks + 1):
                stage.block.append(
                    LTX2AudioResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        norm_type=self.norm_type,
                        causality_axis=self.causality_axis,
                    )
                )
                block_in = block_out
                if self.attn_resolutions:
                    if curr_res in self.attn_resolutions:
                        stage.attn.append(
                            LTX2AudioAttnBlock(
                                block_in, norm_type=self.norm_type
                            )
                        )

            if level != 0:
                stage.upsample = LTX2AudioUpsample(
                    block_in, True, causality_axis=self.causality_axis
                )
                curr_res *= 2

            self.up.insert(0, stage)

        final_block_channels = block_in

        if self.norm_type == "group":
            self.norm_out = nn.GroupNorm(
                num_groups=32,
                num_channels=final_block_channels,
                eps=1e-6,
                affine=True,
            )
        elif self.norm_type == "pixel":
            self.norm_out = LTX2AudioPixelNorm(dim=1, eps=1e-6)
        else:
            raise ValueError(f"Invalid normalization type: {self.norm_type}")

        if self.causality_axis is not None:
            self.conv_out = LTX2AudioCausalConv2d(
                in_channels=final_block_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                causality_axis=self.causality_axis,
            )
        else:
            self.conv_out = nn.Conv2d(
                kernel_size=3,
                in_channels=final_block_channels,
                out_channels=output_channels,
                stride=1,
                padding=1,
                permute=True,
            )

    def input_types(self) -> tuple[TensorType, ...]:
        if self.dtype is None:
            raise ValueError("dtype must be set for input_types")
        if self.device is None:
            raise ValueError("device must be set for input_types")

        # Hardcoded for num_frames=121, frame_rate=24:
        #   latent_channels     = 8
        #   audio_num_frames    = round((121/24) * (16000/160/4)) = 126
        #   latent_mel_bins     = 64 // 4 = 16
        return (
            TensorType(
                self.dtype,
                shape=[1, self.latent_channels, 126, 16],
                device=self.device,
            ),
        )

    def forward(
        self,
        sample: Tensor,
    ) -> Tensor:
        _, _, frames, mel_bins = sample.shape

        target_frames = frames * LATENT_DOWNSAMPLE_FACTOR

        if self.causality_axis is not None:
            # frames >= 1 (positive tensor dim), so frames*4 - 3 >= 1 always.
            # Use Dim arithmetic instead of F.max to keep target_frames as a
            # Dim expression, which the symbolic rmo.slice path requires.
            target_frames = target_frames - (LATENT_DOWNSAMPLE_FACTOR - 1)

        target_channels = self.out_ch
        target_mel_bins = (
            self.mel_bins if self.mel_bins is not None else mel_bins
        )

        hidden_features = self.conv_in(sample)
        hidden_features = self.mid.block_1(hidden_features, temb=None)
        hidden_features = self.mid.attn_1(hidden_features)
        hidden_features = self.mid.block_2(hidden_features, temb=None)

        for level in reversed(range(self.num_resolutions)):
            stage = self.up[level]
            for block_idx, block in enumerate(stage.block):
                hidden_features = block(hidden_features, temb=None)
                if stage.attn:
                    hidden_features = stage.attn[block_idx](hidden_features)

            if level != 0 and hasattr(stage, "upsample"):
                hidden_features = stage.upsample(hidden_features)

        if self.give_pre_end:
            return hidden_features

        hidden = self.norm_out(hidden_features)
        hidden = self.non_linearity(hidden)
        decoded_output = self.conv_out(hidden)
        decoded_output = (
            F.tanh(decoded_output) if self.tanh_out else decoded_output
        )

        target_time = target_frames
        target_freq = target_mel_bins

        # Use a Safe-Pad + Precise-Slice pattern to avoid symbolic Dim
        # comparisons in Python (which throw TypeError). By padding with a
        # small constant (8) that is guaranteed to exceed the maximum
        # architectural discrepancy (2), and then slicing to the exact
        # symbolic target, we produce a result mathematically identical to
        # the original "pad deficit then crop" algorithm.
        decoded_output = F.pad(decoded_output, (0, 0, 0, 0, 0, 8, 0, 8))

        decoded_output = decoded_output[
            :, :target_channels, :target_time, :target_freq
        ]

        return decoded_output


class AutoencoderKLLTX2Audio(nn.Module[[Tensor], Tensor]):
    """Refactored LTX2 Audio Autoencoder (Decode-only)."""

    def __init__(self, config: AutoencoderKLLTX2AudioConfig) -> None:
        super().__init__()
        self.decoder = LTX2AudioDecoder(
            base_channels=config.base_channels,
            output_channels=config.output_channels,
            num_res_blocks=config.num_res_blocks,
            attn_resolutions=config.attn_resolutions,
            in_channels=config.in_channels,
            resolution=config.resolution,
            latent_channels=config.latent_channels,
            ch_mult=config.ch_mult,
            norm_type=config.norm_type,
            causality_axis=config.causality_axis,
            dropout=config.dropout,
            mid_block_add_attention=config.mid_block_add_attention,
            sample_rate=config.sample_rate,
            mel_hop_length=config.mel_hop_length,
            is_causal=config.is_causal,
            mel_bins=config.mel_bins,
            device=config.device,
            dtype=config.dtype,
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.decoder(z)


class AutoencoderKLLTX2AudioModel(BaseAutoencoderModel):
    """ComponentModel wrapper for LTX2 Audio Autoencoder."""

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
            config_class=AutoencoderKLLTX2AudioConfig,
            autoencoder_class=AutoencoderKLLTX2Audio,
        )
