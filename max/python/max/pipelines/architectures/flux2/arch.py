# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from __future__ import annotations

from dataclasses import dataclass

from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.pipelines.core import PixelContext
from max.pipelines.lib import (
    PixelGenerationTokenizer,
    SupportedArchitecture,
    SupportedEncoding,
)
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.interfaces import ArchConfig
from typing_extensions import Self

from .pipeline_flux2 import Flux2Pipeline


@dataclass(kw_only=True)
class Flux2ArchConfig(ArchConfig):
    """Pipeline-level config for Flux2 (implements ArchConfig; no KV cache)."""

    pipeline_config: PipelineConfig

    def get_max_seq_len(self) -> int:
        return 0  # Not used for pixel generation.

    @classmethod
    def initialize(cls, pipeline_config: PipelineConfig) -> Self:
        if len(pipeline_config.model.device_specs) != 1:
            raise ValueError("Flux2 is only supported on a single device")
        return cls(pipeline_config=pipeline_config)


flux2_arch = SupportedArchitecture(
    name="Flux2Pipeline",
    task=PipelineTask.PIXEL_GENERATION,
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={SupportedEncoding.bfloat16: []},
    example_repo_ids=[
        "black-forest-labs/FLUX.2-dev",
    ],
    pipeline_model=Flux2Pipeline,  # type: ignore[arg-type]
    context_type=PixelContext,
    default_weights_format=WeightsFormat.safetensors,
    tokenizer=PixelGenerationTokenizer,
    config=Flux2ArchConfig,
)
