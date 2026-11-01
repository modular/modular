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
"""Configuration for LFM3-VL: the LFM3 decoder plus a vision tower."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import ClassVar

from max.pipelines.lib import MAXModelConfig, PipelineConfig
from max.pipelines.modeling.config_enums import SupportedEncoding
from typing_extensions import Self

from ..lfm3.model_config import Lfm3Config


@dataclass(frozen=True)
class Lfm3VlVisionConfig:
    """The vision tower's shape."""

    hidden_size: int
    intermediate_size: int
    layer_types: list[str]
    num_attention_heads: int
    num_key_value_heads: int
    norm_eps: float
    rope_theta: float
    patch_dim: int
    conv_kernel_size: int
    merge_size: int


@dataclass(kw_only=True)
class Lfm3VlConfig(Lfm3Config):
    """The LFM3 decoder's config, the vision tower's and the projector's."""

    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"bfloat16"}

    vision: Lfm3VlVisionConfig
    projector_hidden_size: int
    projector_bias: bool
    projector_norm_eps: float
    downsample_factor: int

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Builds the config from the decoder's and the vision tower's."""
        text = Lfm3Config.initialize(
            pipeline_config, model_config, max_seq_len=max_seq_len
        )
        hf = (model_config or pipeline_config.model).huggingface_config
        assert hf is not None
        if hf.projector_hidden_act != "gelu":
            raise ValueError(
                f"Unsupported projector activation {hf.projector_hidden_act!r}."
            )
        v = hf.vision_config
        return cls(
            **{f.name: getattr(text, f.name) for f in dataclasses.fields(text)},
            vision=Lfm3VlVisionConfig(
                hidden_size=v.hidden_size,
                intermediate_size=v.intermediate_size,
                layer_types=list(v.layer_types),
                num_attention_heads=v.num_attention_heads,
                num_key_value_heads=v.num_key_value_heads,
                norm_eps=v.norm_eps,
                rope_theta=float(v.rope_parameters["rope_theta"]),
                patch_dim=v.in_channels
                * v.temporal_patch_size
                * v.patch_size**2,
                conv_kernel_size=v.conv_kernel_size,
                merge_size=v.spatial_merge_size,
            ),
            projector_hidden_size=hf.projector_hidden_size,
            projector_bias=hf.projector_bias,
            projector_norm_eps=hf.projector_layer_norm_eps,
            downsample_factor=hf.downsample_factor,
        )
