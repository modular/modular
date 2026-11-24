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
"""Config for d1's LFM3-VL and LFM3-Omni backbones."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import ClassVar

from max.pipelines.lib import MAXModelConfig, PipelineConfig
from max.pipelines.modeling.config_enums import SupportedEncoding
from typing_extensions import Self

from ..lfm3_audio.model_config import Lfm3AudioEncoderConfig
from ..lfm3_vl.model_config import Lfm3VlConfig


@dataclass(kw_only=True)
class D1Config(Lfm3VlConfig):
    """A d1 backbone's config.

    LFM3-VL's config, and the speech encoder's when the checkpoint hears.
    """

    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"bfloat16"}

    encoder: Lfm3AudioEncoderConfig | None = None

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Builds the config, rejecting a checkpoint not trained on sets."""
        vl = Lfm3VlConfig.initialize(
            pipeline_config, model_config, max_seq_len=max_seq_len
        )
        hf = (model_config or pipeline_config.model).huggingface_config
        assert hf is not None
        if getattr(hf, "d1_attention", None) != "set":
            raise ValueError("MAX serves d1 checkpoints trained on sets.")
        audio = getattr(hf, "audio_encoder_config", None)
        return cls(
            **{f.name: getattr(vl, f.name) for f in dataclasses.fields(vl)},
            encoder=None
            if audio is None
            else Lfm3AudioEncoderConfig.from_hf(audio),
        )
