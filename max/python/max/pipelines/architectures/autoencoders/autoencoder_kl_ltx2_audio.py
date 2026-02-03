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

from typing import Any, ClassVar

from max import functional as F
from max.driver import Device
from max.graph.weights import Weights
from max.nn import (
    Conv2d,
    Dropout,
    GroupNorm,
    Identity,
    Linear,
    Module,
    ModuleList,
    SiLU,
)
from max.pipelines.lib import SupportedEncoding
from max.tensor import Tensor

from .model import BaseAutoencoderModel
from .model_config import AutoencoderKLLTX2AudioConfig

LATENT_DOWNSAMPLE_FACTOR = 4


class LTX2AudioCausalConv2d(Module[[Tensor], Tensor]):
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
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
            )
        elif self.causality_axis in {"width", "width-compatibility"}:
            self.padding = (pad_w, 0, pad_h // 2, pad_h - pad_h // 2)
        elif self.causality_axis == "height":
            self.padding = (pad_w // 2, pad_w - pad_w // 2, pad_h, 0)
        else:
            raise ValueError(f"Invalid causality_axis: {causality_axis}")

        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            num_groups=groups,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, self.padding)
        return self.conv(x)


class LTX2AudioPixelNorm(Module[[Tensor], Tensor]):
    """Per-pixel (per-location) RMS normalization layer."""

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean_sq = F.mean(x**2, axis=self.dim, keepdims=True)
        rms = F.sqrt(mean_sq + self.eps)
        return x / rms


class LTX2AudioAttnBlock(Module[[Tensor], Tensor]):
    """Attention block for LTX2 Audio."""

    def __init__(
        self,
        in_channels: int,
        norm_type: str = "group",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        if norm_type == "group":
            self.norm = GroupNorm(
                num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
            )
        elif norm_type == "pixel":
            self.norm = LTX2AudioPixelNorm(dim=1, eps=1e-6)
        else:
            raise ValueError(f"Invalid normalization type: {norm_type}")

        self.q = Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        batch, channels, height, width = q.shape
        # Use F.bmm for attention computation
        q = F.reshape(q, [batch, channels, height * width]).permute(
            0, 2, 1
        )  # [B, HW, C]
        k = F.reshape(k, [batch, channels, height * width])  # [B, C, HW]

        attn = F.bmm(q, k) * (float(channels) ** (-0.5))  # [B, HW, HW]
        attn = F.softmax(attn, axis=2)

        v = F.reshape(v, [batch, channels, height * width])  # [B, C, HW]
        attn = attn.permute(0, 2, 1)  # [B, HW, HW]

        h = F.bmm(v, attn)  # [B, C, HW]
        h = F.reshape(h, [batch, channels, height, width])

        return x + self.proj_out(h)


class LTX2AudioResnetBlock(Module[[Tensor, Tensor | None], Tensor]):
    """ResNet block for LTX2 Audio."""

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
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        if norm_type == "group":
            self.norm1 = GroupNorm(
                num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
            )
        elif norm_type == "pixel":
            self.norm1 = LTX2AudioPixelNorm(dim=1, eps=1e-6)
        else:
            raise ValueError(f"Invalid normalization type: {norm_type}")

        self.non_linearity = SiLU()

        if causality_axis is not None and causality_axis != "none":
            self.conv1 = LTX2AudioCausalConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                causality_axis=causality_axis,
            )
        else:
            self.conv1 = Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1
            )

        self.temb_proj: Linear | None = None
        if temb_channels > 0:
            self.temb_proj = Linear(temb_channels, out_channels)

        if norm_type == "group":
            self.norm2 = GroupNorm(
                num_groups=32, num_channels=out_channels, eps=1e-6, affine=True
            )
        elif norm_type == "pixel":
            self.norm2 = LTX2AudioPixelNorm(dim=1, eps=1e-6)

        self.dropout = Dropout(dropout)

        if causality_axis is not None and causality_axis != "none":
            self.conv2 = LTX2AudioCausalConv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                causality_axis=causality_axis,
            )
        else:
            self.conv2 = Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1
            )

        self.nin_shortcut: Module[[Tensor], Tensor] | None = None
        self.conv_shortcut: Module[[Tensor], Tensor] | None = None
        if in_channels != out_channels:
            if conv_shortcut:
                if causality_axis is not None and causality_axis != "none":
                    self.conv_shortcut = LTX2AudioCausalConv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        causality_axis=causality_axis,
                    )
                else:
                    self.conv_shortcut = Conv2d(
                        in_channels, out_channels, kernel_size=3, padding=1
                    )
            else:
                if causality_axis is not None and causality_axis != "none":
                    self.nin_shortcut = LTX2AudioCausalConv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        causality_axis=causality_axis,
                    )
                else:
                    self.nin_shortcut = Conv2d(
                        in_channels, out_channels, kernel_size=1
                    )

    def forward(self, x: Tensor, temb: Tensor | None = None) -> Tensor:
        h = self.norm1(x)
        h = self.non_linearity(h)
        h = self.conv1(h)

        if temb is not None and self.temb_proj is not None:
            h = h + self.temb_proj(self.non_linearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.non_linearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        residual = x
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(x)
            else:
                residual = self.nin_shortcut(x)

        return residual + h


class LTX2AudioUpsample(Module[[Tensor], Tensor]):
    """Upsampling block for LTX2 Audio."""

    def __init__(
        self,
        in_channels: int,
        with_conv: bool,
        causality_axis: str | None = "height",
    ) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis

        self.conv: Module[[Tensor], Tensor] | None = None
        if self.with_conv:
            if causality_axis is not None and causality_axis != "none":
                self.conv = LTX2AudioCausalConv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    causality_axis=causality_axis,
                )
            else:
                self.conv = Conv2d(
                    in_channels, in_channels, kernel_size=3, padding=1
                )

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.conv is not None:
            x = self.conv(x)
            if self.causality_axis == "height":
                x = x[:, :, 1:, :]
            elif self.causality_axis == "width":
                x = x[:, :, :, 1:]
        return x


class LTX2AudioAudioPatchifier:
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
        return F.reshape(
            audio_latents.permute(0, 2, 1, 3), [batch, time, channels * freq]
        )

    def unpatchify(
        self, audio_latents: Tensor, channels: int, mel_bins: int
    ) -> Tensor:
        batch, time, _ = audio_latents.shape
        return F.reshape(
            audio_latents, [batch, time, channels, mel_bins]
        ).permute(0, 2, 1, 3)


class LTX2AudioDecoder(Module[[Tensor], Tensor]):
    """LTX2 Audio Decoder module."""

    def __init__(self, config: AutoencoderKLLTX2AudioConfig) -> None:
        super().__init__()
        self.config = config

        self.patchifier = LTX2AudioAudioPatchifier(
            patch_size=1,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            sample_rate=config.sample_rate,
            hop_length=config.mel_hop_length,
            is_causal=config.is_causal,
        )

        curr_channels = config.base_channels * config.ch_mult[-1]

        if (
            config.causality_axis is not None
            and config.causality_axis != "none"
        ):
            self.conv_in = LTX2AudioCausalConv2d(
                config.latent_channels,
                curr_channels,
                kernel_size=3,
                causality_axis=config.causality_axis,
            )
        else:
            self.conv_in = Conv2d(
                config.latent_channels, curr_channels, kernel_size=3, padding=1
            )

        self.mid = Module()
        self.mid.block_1 = LTX2AudioResnetBlock(
            in_channels=curr_channels,
            out_channels=curr_channels,
            temb_channels=0,
            dropout=config.dropout,
            norm_type=config.norm_type,
            causality_axis=config.causality_axis,
        )

        if config.mid_block_add_attention:
            self.mid.attn_1 = LTX2AudioAttnBlock(
                curr_channels, norm_type=config.norm_type
            )
        else:
            self.mid.attn_1 = Identity()

        self.mid.block_2 = LTX2AudioResnetBlock(
            in_channels=curr_channels,
            out_channels=curr_channels,
            temb_channels=0,
            dropout=config.dropout,
            norm_type=config.norm_type,
            causality_axis=config.causality_axis,
        )

        self.up_blocks = ModuleList()
        for level in reversed(range(len(config.ch_mult))):
            block_out = config.base_channels * config.ch_mult[level]

            stage = Module()
            stage.resnets = ModuleList()
            stage.attns = ModuleList()

            for _ in range(config.num_res_blocks + 1):
                stage.resnets.append(
                    LTX2AudioResnetBlock(
                        in_channels=curr_channels,
                        out_channels=block_out,
                        temb_channels=0,
                        dropout=config.dropout,
                        norm_type=config.norm_type,
                        causality_axis=config.causality_axis,
                    )
                )
                curr_channels = block_out
                # For simplicity, we'll omit complex attention resolution logic unless needed
                # (diffusers typically adds attention at fixed resolutions)

            if level != 0:
                stage.upsample = LTX2AudioUpsample(
                    curr_channels, True, causality_axis=config.causality_axis
                )

            self.up_blocks.append(stage)

        if config.norm_type == "group":
            self.norm_out = GroupNorm(
                num_groups=32, num_channels=curr_channels, eps=1e-6, affine=True
            )
        elif config.norm_type == "pixel":
            self.norm_out = LTX2AudioPixelNorm(dim=1, eps=1e-6)
        else:
            raise ValueError(f"Invalid normalization type: {config.norm_type}")

        if (
            config.causality_axis is not None
            and config.causality_axis != "none"
        ):
            self.conv_out = LTX2AudioCausalConv2d(
                curr_channels,
                config.output_channels,
                kernel_size=3,
                causality_axis=config.causality_axis,
            )
        else:
            self.conv_out = Conv2d(
                curr_channels, config.output_channels, kernel_size=3, padding=1
            )

    def forward(self, sample: Tensor) -> Tensor:
        h = self.conv_in(sample)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        for stage in self.up_blocks:
            for resnet in stage.resnets:
                h = resnet(h)
            if hasattr(stage, "upsample"):
                h = stage.upsample(h)

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        # In actual LTX2 Audio, there's complex padding/cropping based on target audio length
        # For the MAX model, we return the raw decoded output
        return h


class AutoencoderKLLTX2Audio(Module[[Tensor], Tensor]):
    """Refactored LTX2 Audio Autoencoder (Decode-only)."""

    def __init__(self, config: AutoencoderKLLTX2AudioConfig) -> None:
        super().__init__()
        self.config = config
        self.decoder = LTX2AudioDecoder(config)

    def forward(self, z: Tensor) -> Tensor:
        return self.decoder(z)


class AutoencoderKLLTX2AudioModel(BaseAutoencoderModel):
    """MaxModel wrapper for LTX2 Audio Autoencoder."""

    config_name: ClassVar[str] = AutoencoderKLLTX2AudioConfig.config_name

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
