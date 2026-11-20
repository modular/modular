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
"""Configuration for LFM3-Audio: the LFM3 decoder plus a speech encoder."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import ClassVar

from max.pipelines.lib import MAXModelConfig, PipelineConfig
from max.pipelines.modeling.config_enums import SupportedEncoding
from transformers import GenerationConfig, PretrainedConfig
from typing_extensions import Self

from ..lfm3.model_config import CONV, SLIDING_ATTENTION, Lfm3Config

CODEC_SUBFOLDER = "audio_tokenizer"
"""Where a checkpoint keeps its LiquidCodec, which turns frames into
samples."""


@dataclass(frozen=True, kw_only=True)
class Lfm3StackConfig:
    """The shape of a cacheless LFM stack.

    Its attention reads a band of ``sliding_window`` rows.
    """

    hidden_size: int
    intermediate_size: int
    layer_types: list[str]
    num_attention_heads: int
    num_key_value_heads: int
    norm_eps: float
    rope_theta: float
    conv_kernel_size: int
    sliding_window: int
    use_sandwich_norm: bool = False
    gated_attention: bool = False
    """Whether attention has LFM3's per-head sigmoid output gate."""

    @classmethod
    def from_lfm3(cls, lfm3: PretrainedConfig, *, sliding_window: int) -> Self:
        """Builds the stack shape from an LFM3 config and a band width."""
        (rope,) = lfm3.rope_parameters.values()
        return cls(
            hidden_size=lfm3.hidden_size,
            intermediate_size=lfm3.intermediate_size,
            layer_types=list(lfm3.layer_types),
            num_attention_heads=lfm3.num_attention_heads,
            num_key_value_heads=lfm3.num_key_value_heads,
            norm_eps=lfm3.norm_eps,
            rope_theta=float(rope["rope_theta"]),
            conv_kernel_size=lfm3.conv_L_cache,
            sliding_window=sliding_window,
            use_sandwich_norm=lfm3.use_sandwich_norm,
            gated_attention=True,
        )


@dataclass(frozen=True, kw_only=True)
class Lfm3AudioEncoderConfig(Lfm3StackConfig):
    """The speech encoder's shape: the downsampler's and its LFM2 stack's."""

    num_mel_bins: int
    conv_channels: int

    @classmethod
    def from_hf(cls, encoder: PretrainedConfig) -> Self:
        """Builds the encoder config from the checkpoint's encoder config."""
        lfm2 = encoder.text_config
        intermediate = lfm2.intermediate_size
        if lfm2.block_auto_adjust_ff_dim:
            intermediate = int(2 * intermediate / 3)
            if lfm2.block_ffn_dim_multiplier is not None:
                intermediate = int(lfm2.block_ffn_dim_multiplier * intermediate)
            multiple = lfm2.block_multiple_of
            intermediate = multiple * -(intermediate // -multiple)
        return cls(
            num_mel_bins=encoder.num_mel_bins,
            conv_channels=encoder.conv_channels,
            hidden_size=encoder.hidden_size,
            intermediate_size=intermediate,
            layer_types=list(lfm2.layer_types),
            num_attention_heads=lfm2.num_attention_heads,
            num_key_value_heads=lfm2.num_key_value_heads,
            norm_eps=lfm2.norm_eps,
            rope_theta=float(lfm2.rope_parameters["rope_theta"]),
            conv_kernel_size=lfm2.conv_L_cache,
            sliding_window=encoder.sliding_window,
        )


@dataclass(frozen=True, kw_only=True)
class Lfm3SpeechConfig:
    """What sampling speech frames takes.

    The depth decoder's shape, the frame layout, the mode tokens and the
    sampling settings.
    """

    depth: Lfm3StackConfig
    num_codebooks: int
    codebook_size: int
    audio_eos_code: int
    maskgit_schedule: list[int]
    interleaved_n_text: int
    interleaved_n_audio: int
    audio_out_token_id: int
    audio_start_token_id: int
    mixed_start_token_id: int
    text_end_token_id: int
    end_token_id: int
    """``<|im_end|>``, which ends the reply."""
    temperature: float | None
    top_k: int | None
    repetition_penalty: float
    repetition_window: int

    @classmethod
    def from_hf(
        cls, hf: PretrainedConfig, generation: GenerationConfig
    ) -> Self:
        """Reads the checkpoint's config and its generation config."""
        return cls(
            depth=Lfm3StackConfig.from_lfm3(
                hf.depth_config, sliding_window=hf.num_codebooks
            ),
            num_codebooks=hf.num_codebooks,
            codebook_size=hf.codebook_size,
            audio_eos_code=hf.audio_eos_code,
            maskgit_schedule=list(hf.maskgit_schedule),
            interleaved_n_text=hf.interleaved_n_text,
            interleaved_n_audio=hf.interleaved_n_audio,
            audio_out_token_id=hf.audio_out_token_id,
            audio_start_token_id=hf.audio_start_token_id,
            mixed_start_token_id=hf.mixed_start_token_id,
            text_end_token_id=hf.text_end_token_id,
            end_token_id=hf.text_config.eos_token_id,
            temperature=getattr(generation, "audio_temperature", None),
            top_k=getattr(generation, "audio_top_k", None),
            repetition_penalty=getattr(
                generation, "audio_repetition_penalty", None
            )
            or 1.0,
            repetition_window=getattr(
                generation, "audio_repetition_window", None
            )
            or 64,
        )


@dataclass(frozen=True, kw_only=True)
class Lfm3CodecConfig:
    """The LiquidCodec decoder's shape: frame codes to 24 kHz samples."""

    decoder: Lfm3StackConfig
    num_codebooks: int
    fsq_levels: list[int]
    quantizer_dim: int
    n_fft: int
    hop_length: int
    frame_stack: int
    """The spectrogram frames per codec frame."""
    sample_rate: int

    @classmethod
    def from_hf(cls, codec: PretrainedConfig) -> Self:
        """Builds the codec decoder config from the checkpoint's."""
        # LiquidCodecConfig.lfm_config fixes the stack rather than exporting it.
        decoder = Lfm3StackConfig(
            hidden_size=codec.encoder_dim,
            intermediate_size=2304,
            layer_types=[
                SLIDING_ATTENTION if i in (2, 4, 6) else CONV for i in range(8)
            ],
            num_attention_heads=8,
            num_key_value_heads=2,
            norm_eps=1e-5,
            rope_theta=1_000_000.0,
            conv_kernel_size=7,
            sliding_window=codec.decoder_window,
            use_sandwich_norm=True,
            gated_attention=True,
        )
        return cls(
            decoder=decoder,
            num_codebooks=codec.n_q,
            fsq_levels=list(codec.fsq_levels),
            quantizer_dim=codec.quantizer_dim,
            n_fft=codec.stft_n_fft,
            hop_length=codec.stft_hop_length,
            frame_stack=codec.stft_frame_stack,
            sample_rate=codec.sample_rate,
        )

    @property
    def context_frames(self) -> int:
        """The codec frames before a frame that its samples depend on."""
        stack = self.decoder
        reach = sum(
            stack.conv_kernel_size - 1
            if kind == CONV
            else stack.sliding_window - 1
            for kind in stack.layer_types
        )
        return -(reach // -self.frame_stack)


@dataclass(kw_only=True)
class Lfm3AudioConfig(Lfm3Config):
    """The LFM3 decoder's config, the speech encoder's and the projector's."""

    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"bfloat16"}

    encoder: Lfm3AudioEncoderConfig
    speech: Lfm3SpeechConfig
    codec: Lfm3CodecConfig
    suppressed_token_ids: list[int]
    """The placeholders and the speech-mode switches, which text output
    never samples."""

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Builds the config from the decoder's, encoder's and codec's."""
        text = Lfm3Config.initialize(
            pipeline_config, model_config, max_seq_len=max_seq_len
        )
        model = model_config or pipeline_config.model
        hf = model.huggingface_config
        assert hf is not None
        repo = model.huggingface_model_repo
        codec = PretrainedConfig.from_pretrained(
            repo.repo_id, subfolder=CODEC_SUBFOLDER, revision=repo.revision
        )
        return cls(
            **{f.name: getattr(text, f.name) for f in dataclasses.fields(text)},
            encoder=Lfm3AudioEncoderConfig.from_hf(hf.audio_encoder_config),
            speech=Lfm3SpeechConfig.from_hf(hf, model.generation_config),
            codec=Lfm3CodecConfig.from_hf(codec),
            suppressed_token_ids=[
                hf.audio_token_id,
                hf.audio_out_token_id,
                hf.audio_start_token_id,
                hf.mixed_start_token_id,
            ],
        )
