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

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import max.experimental.functional as F
from max.driver import Device
from max.dtype import DType
from max.experimental import nn
from max.experimental.nn.common_layers.activation import (
    activation_function_from_name,
)
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
        dtype: DType | None = None,
    ) -> None:
        super().__init__()
        self.causality_axis = causality_axis
        self._dtype = dtype
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
            dtype=self._dtype,
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
        dtype: DType | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.norm: nn.Module[[Tensor], Tensor]

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
            dtype=dtype,
        )
        self.k = nn.Conv2d(
            kernel_size=1,
            in_channels=in_channels,
            out_channels=in_channels,
            permute=True,
            dtype=dtype,
        )
        self.v = nn.Conv2d(
            kernel_size=1,
            in_channels=in_channels,
            out_channels=in_channels,
            permute=True,
            dtype=dtype,
        )
        self.proj_out = nn.Conv2d(
            kernel_size=1,
            in_channels=in_channels,
            out_channels=in_channels,
            permute=True,
            dtype=dtype,
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        batch, channels, height, width = q.shape
        q = q.reshape((batch, channels, height * width)).permute([0, 2, 1])
        k = k.reshape((batch, channels, height * width))

        attn = F.matmul(q, k) * (int(channels) ** (-0.5))
        attn = F.softmax(attn, axis=-1)

        v = v.reshape((batch, channels, height * width))
        attn = attn.permute([0, 2, 1])

        h = F.matmul(v, attn).reshape((batch, channels, height, width))

        return x + self.proj_out(h)


class LTX2AudioResnetBlock(nn.Module[..., Tensor]):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        norm_type: str = "group",
        causality_axis: str | None = "height",
        dtype: DType | None = None,
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

        self.norm1: nn.Module[[Tensor], Tensor]
        if norm_type == "group":
            self.norm1 = nn.GroupNorm(
                num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
            )
        elif norm_type == "pixel":
            self.norm1 = LTX2AudioPixelNorm(dim=1, eps=1e-6)
        else:
            raise ValueError(f"Invalid normalization type: {norm_type}")
        self.non_linearity = activation_function_from_name("silu")
        self.conv1: nn.Module[[Tensor], Tensor]
        if causality_axis is not None:
            self.conv1 = LTX2AudioCausalConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                causality_axis=causality_axis,
                dtype=dtype,
            )
        else:
            self.conv1 = nn.Conv2d(
                kernel_size=3,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=1,
                padding=1,
                permute=True,
                dtype=dtype,
            )
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2: nn.Module[[Tensor], Tensor]
        if norm_type == "group":
            self.norm2 = nn.GroupNorm(
                num_groups=32, num_channels=out_channels, eps=1e-6, affine=True
            )
        elif norm_type == "pixel":
            self.norm2 = LTX2AudioPixelNorm(dim=1, eps=1e-6)
        else:
            raise ValueError(f"Invalid normalization type: {norm_type}")
        self.dropout = nn.Dropout(dropout)
        self.conv2: nn.Module[[Tensor], Tensor]
        if causality_axis is not None:
            self.conv2 = LTX2AudioCausalConv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                causality_axis=causality_axis,
                dtype=dtype,
            )
        else:
            self.conv2 = nn.Conv2d(
                kernel_size=3,
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                padding=1,
                permute=True,
                dtype=dtype,
            )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut: nn.Module[[Tensor], Tensor]
                if causality_axis is not None:
                    self.conv_shortcut = LTX2AudioCausalConv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        causality_axis=causality_axis,
                        dtype=dtype,
                    )
                else:
                    self.conv_shortcut = nn.Conv2d(
                        kernel_size=3,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=1,
                        padding=1,
                        permute=True,
                        dtype=dtype,
                    )
            else:
                self.nin_shortcut: nn.Module[[Tensor], Tensor]
                if causality_axis is not None:
                    self.nin_shortcut = LTX2AudioCausalConv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        causality_axis=causality_axis,
                        dtype=dtype,
                    )
                else:
                    self.nin_shortcut = nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        permute=True,
                        dtype=dtype,
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
        dtype: DType | None = None,
    ) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis
        self.conv: nn.Module[[Tensor], Tensor]
        if self.with_conv:
            if causality_axis is not None:
                self.conv = LTX2AudioCausalConv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    causality_axis=causality_axis,
                    dtype=dtype,
                )
            else:
                self.conv = nn.Conv2d(
                    kernel_size=3,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    stride=1,
                    padding=1,
                    permute=True,
                    dtype=dtype,
                )

    def forward(self, x: Tensor) -> Tensor:
        x = interpolate_2d_nearest(x, scale_factor=2)  # type: ignore[assignment]
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
        return audio_latents.permute([0, 2, 1, 3]).reshape(
            (batch, time, channels * freq)
        )

    def unpatchify(
        self, audio_latents: Tensor, channels: int, mel_bins: int
    ) -> Tensor:
        batch, time, _ = audio_latents.shape
        return audio_latents.reshape((batch, time, channels, mel_bins)).permute(
            [0, 2, 1, 3]
        )

    @property
    def patch_size(self) -> tuple[int, int, int]:
        return self._patch_size


class LTX2AudioDecoderMid(nn.Module[..., Any]):
    """Container for the middle block of the LTX2 Audio Decoder."""

    block_1: LTX2AudioResnetBlock
    attn_1: nn.Module[[Tensor], Tensor]
    block_2: LTX2AudioResnetBlock


class LTX2AudioDecoderStage(nn.Module[..., Any]):
    """Container for a single stage (level) of the LTX2 Audio Decoder."""

    block: nn.ModuleList[LTX2AudioResnetBlock]
    attn: nn.ModuleList[LTX2AudioAttnBlock]
    upsample: LTX2AudioUpsample


class LTX2AudioDecoder(nn.Module[[Tensor], Tensor]):
    """
    Symmetric decoder that reconstructs audio spectrograms from latent features.

    The decoder mirrors the encoder structure with configurable channel multipliers, attention resolutions, and causal
    convolutions.
    """

    def __init__(
        self,
        base_channels: int = 128,
        out_channels: int = 1,
        num_res_blocks: int = 2,
        attn_resolutions: list[int] | None = None,
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
        self.out_ch = out_channels
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

        self.conv_in: nn.Module[[Tensor], Tensor]
        if self.causality_axis is not None:
            self.conv_in = LTX2AudioCausalConv2d(
                latent_channels,
                base_block_channels,
                kernel_size=3,
                stride=1,
                causality_axis=self.causality_axis,
                dtype=self.dtype,
            )
        else:
            self.conv_in = nn.Conv2d(
                kernel_size=3,
                in_channels=latent_channels,
                out_channels=base_block_channels,
                stride=1,
                padding=1,
                permute=True,
                dtype=self.dtype,
            )
        self.non_linearity = activation_function_from_name("silu")
        self.mid = LTX2AudioDecoderMid()
        self.mid.block_1 = LTX2AudioResnetBlock(
            in_channels=base_block_channels,
            out_channels=base_block_channels,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            dtype=self.dtype,
        )
        if mid_block_add_attention:
            self.mid.attn_1 = LTX2AudioAttnBlock(
                base_block_channels, norm_type=self.norm_type, dtype=self.dtype
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
            dtype=self.dtype,
        )

        self.up: nn.ModuleList[LTX2AudioDecoderStage] = nn.ModuleList()
        block_in = base_block_channels
        curr_res = self.resolution // (2 ** (self.num_resolutions - 1))

        for level in reversed(range(self.num_resolutions)):
            stage = LTX2AudioDecoderStage()
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
                        dtype=self.dtype,
                    )
                )
                block_in = block_out
                if self.attn_resolutions:
                    if curr_res in self.attn_resolutions:
                        stage.attn.append(
                            LTX2AudioAttnBlock(
                                block_in,
                                norm_type=self.norm_type,
                                dtype=self.dtype,
                            )
                        )

            if level != 0:
                stage.upsample = LTX2AudioUpsample(
                    block_in,
                    True,
                    causality_axis=self.causality_axis,
                    dtype=self.dtype,
                )
                curr_res *= 2

            self.up.insert(0, stage)

        final_block_channels = block_in

        self.norm_out: nn.Module[[Tensor], Tensor]
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

        self.conv_out: nn.Module[[Tensor], Tensor]
        if self.causality_axis is not None:
            self.conv_out = LTX2AudioCausalConv2d(
                in_channels=final_block_channels,
                out_channels=self.out_ch,
                kernel_size=3,
                stride=1,
                causality_axis=self.causality_axis,
                dtype=self.dtype,
            )
        else:
            self.conv_out = nn.Conv2d(
                kernel_size=3,
                in_channels=final_block_channels,
                out_channels=self.out_ch,
                stride=1,
                padding=1,
                permute=True,
                dtype=self.dtype,
            )

    def input_types(self) -> tuple[TensorType, ...]:
        """Returns the expected input types for the decoder."""
        return (
            TensorType(
                self.dtype if self.dtype is not None else DType.bfloat16,
                shape=[
                    "batch_size",
                    self.latent_channels,
                    "audio_num_frames",
                    "num_mel_bins",
                ],
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
            out_channels=config.out_channels,
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


class PostprocessAndDecode(nn.Module[..., Tensor]):
    """Fused latent denorm + unpack + VAE decode in a single compiled graph.

    Eliminates the inter-graph boundary and intermediate tensor materialization
    by fusing the _denormalize_audio_latents, _unpack_audio_latents, and
    audio_vae.decode steps from the diffusers pipeline.

    Accepts patchified latents in (B, T, D) shape, where
    D = latent_channels * latent_mel_bins (e.g. 8 * 16 = 128).
    """

    def __init__(
        self,
        decoder: LTX2AudioDecoder,
        latents_mean: Tensor,
        latents_std: Tensor,
        num_channels: int,
        latent_mel_bins: int,
        device: DeviceRef,
        dtype: DType,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.latents_mean = latents_mean
        self.latents_std = latents_std
        self._num_channels = num_channels
        self._latent_mel_bins = latent_mel_bins
        self._device = device
        self._dtype = dtype

    def forward(
        self,
        latents_btd: Tensor,
    ) -> Tensor:
        """Run latent denorm + unpack + VAE decode in one fused graph.

        Args:
            latents_btd: Normalized patchified latent tensor of shape (B, T, D).

        Returns:
            Decoded mel-spectrogram tensor.
        """
        # Denormalization: (B, T, D) * (D,) → element-wise broadcast on last dim
        latents_btd = latents_btd * self.latents_std + self.latents_mean

        # Unpack patchified (B, T, D) → spatial (B, C, T, M)
        # D = C * M, where C = latent_channels, M = latent_mel_bins
        batch = latents_btd.shape[0]
        t = latents_btd.shape[1]
        latent_channels = self._num_channels // self._latent_mel_bins
        latents = F.reshape(
            latents_btd, (batch, t, latent_channels, self._latent_mel_bins)
        )
        latents = F.permute(
            latents, (0, 2, 1, 3)
        )  # (B, T, C, M) → (B, C, T, M)

        return self.decoder(latents)

    def input_types(self) -> tuple[TensorType, ...]:
        return (
            TensorType(
                self._dtype,
                shape=["batch_size", "audio_num_frames", self._num_channels],
                device=self._device,
            ),
        )


class AutoencoderKLLTX2AudioModel(BaseAutoencoderModel):
    """ComponentModel wrapper for AutoencoderKLLTX2Audio.

    This class provides the ComponentModel interface for AutoencoderKLLTX2Audio,
    handling configuration, weight loading, model compilation, and BatchNorm
    statistics for LTX2Audio's latent patchification.
    """

    latents_mean: Tensor
    latents_std: Tensor

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        """Initialize AutoencoderKLLTX2AudioModel.

        Args:
            config: Model configuration dictionary.
            encoding: Supported encoding for the model.
            devices: List of devices to use.
            weights: Model weights.
        """
        super().__init__(
            config=config,
            encoding=encoding,
            devices=devices,
            weights=weights,
            config_class=AutoencoderKLLTX2AudioConfig,
            autoencoder_class=AutoencoderKLLTX2Audio,
        )

    def load_model(self) -> Any:
        """Load and compile the decoder and encoder models with BatchNorm statistics.

        Extracts BatchNorm statistics (latents_mean, latents_std) which are specific to LTX2, then
        delegates to base class for weight loading and model compilation.

        Returns:
            Compiled decoder model callable.
        """
        bn_stats = {}

        for key, value in self.weights.items():
            if key in ("latents_mean", "latents_std"):
                weight_data = value.data()
                target_dtype = self.config.dtype
                if weight_data.dtype != target_dtype:
                    if weight_data.dtype.is_float() and target_dtype.is_float():
                        weight_data = weight_data.astype(target_dtype)
                    # Non-float left as-is; running_mean/var are typically float.
                bn_stats[key] = weight_data.data

        bn_mean_data = bn_stats.get("latents_mean")
        bn_std_data = bn_stats.get("latents_std")

        if bn_mean_data is None or bn_std_data is None:
            raise ValueError(
                "Latents statistics (latents_mean, latents_std) not loaded. "
                "Make sure the model weights contain 'latents_mean' and 'latents_std'."
            )

        super().load_model()

        self.latents_mean = Tensor.from_dlpack(bn_mean_data).to(self.devices[0])
        self.latents_std = Tensor.from_dlpack(bn_std_data).to(self.devices[0])

        return self.model

    def build_fused_decode(
        self, device: Device, num_channels: int, latent_mel_bins: int
    ) -> Callable[..., Any]:
        """Build a fused postprocess + VAE decode compiled graph.

        Combines latent denormalization, unpacking from patchified (B, T, D)
        to spatial (B, C, T, M), and VAE decoding into a single compiled graph.

        Args:
            device: Target device for the compiled graph.
            num_channels: Patchified latent feature size D = latent_channels *
                latent_mel_bins (equals latents_mean.shape[0]).
            latent_mel_bins: Number of mel bins in the latent space
                (mel_bins // LATENT_DOWNSAMPLE_FACTOR, typically 16).

        Returns:
            Compiled callable taking a patchified latent tensor (B, T, D) and
            returning the decoded mel-spectrogram tensor.
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
            autoencoder = AutoencoderKLLTX2Audio(self.config)
            fused = PostprocessAndDecode(
                decoder=autoencoder.decoder,
                latents_mean=self.latents_mean,
                latents_std=self.latents_std,
                num_channels=num_channels,
                latent_mel_bins=latent_mel_bins,
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
        """Property to access BatchNorm statistics, compatible with diffusers API.

        Returns a SimpleNamespace with running_mean and running_var attributes
        for compatibility with pipeline code that accesses self.vae.bn.running_mean.

        Returns:
            SimpleNamespace: Object containing running_mean and running_var attributes.
        """
        return SimpleNamespace(
            running_mean=self.latents_mean, running_var=self.latents_std
        )
