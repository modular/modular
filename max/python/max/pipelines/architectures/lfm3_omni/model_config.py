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
"""Configuration for LFM3-Omni: LFM3-VL's and LFM3-Audio's, combined."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import ClassVar

from max.pipelines.lib import MAXModelConfig, PipelineConfig
from max.pipelines.modeling.config_enums import SupportedEncoding
from typing_extensions import Self

from ..lfm3_audio.model_config import Lfm3AudioConfig
from ..lfm3_vl.model_config import Lfm3VlConfig


@dataclass(kw_only=True)
class Lfm3OmniConfig(Lfm3VlConfig, Lfm3AudioConfig):
    """The LFM3-Omni config.

    The LFM3 decoder's config, the vision tower's and the speech encoder's.
    """

    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"bfloat16"}

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Builds the config from LFM3-VL's and LFM3-Audio's."""
        parts = (
            base.initialize(
                pipeline_config, model_config, max_seq_len=max_seq_len
            )
            for base in (Lfm3VlConfig, Lfm3AudioConfig)
        )
        return cls(
            **{
                f.name: getattr(part, f.name)
                for part in parts
                for f in dataclasses.fields(part)
            }
        )
