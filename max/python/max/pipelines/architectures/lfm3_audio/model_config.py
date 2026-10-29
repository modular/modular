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
from transformers import PretrainedConfig
from typing_extensions import Self

from ..lfm3.model_config import Lfm3Config


@dataclass(frozen=True)
class Lfm3AudioEncoderConfig:
    """The speech encoder's shape: the downsampler's and its LFM2 stack's."""

    num_mel_bins: int
    conv_channels: int
    hidden_size: int
    intermediate_size: int
    layer_types: list[str]
    num_attention_heads: int
    num_key_value_heads: int
    norm_eps: float
    rope_theta: float
    conv_kernel_size: int
    sliding_window: int

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


@dataclass(kw_only=True)
class Lfm3AudioConfig(Lfm3Config):
    """The LFM3 decoder's config, the speech encoder's and the projector's."""

    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"bfloat16"}

    encoder: Lfm3AudioEncoderConfig
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
        hf = (model_config or pipeline_config.model).huggingface_config
        assert hf is not None
        return cls(
            **{f.name: getattr(text, f.name) for f in dataclasses.fields(text)},
            encoder=Lfm3AudioEncoderConfig.from_hf(hf.audio_encoder_config),
            suppressed_token_ids=[
                hf.audio_token_id,
                hf.audio_out_token_id,
                hf.audio_start_token_id,
                hf.mixed_start_token_id,
            ],
        )
