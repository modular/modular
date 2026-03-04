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

from dataclasses import dataclass

from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.pipelines.core import PixelContext
from max.pipelines.lib import (
    PixelGenerationTokenizer,
    SupportedArchitecture,
)
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.interfaces import ArchConfig
from typing_extensions import Self

from .pipeline_flux import FluxPipeline


@dataclass(kw_only=True)
class FluxArchConfig(ArchConfig):
    """Pipeline-level config for Flux1 (implements ArchConfig; no KV cache)."""

    pipeline_config: PipelineConfig

    def get_max_seq_len(self) -> int:
        return 77

    @classmethod
    def initialize(cls, pipeline_config: PipelineConfig) -> Self:
        return cls(pipeline_config=pipeline_config)


flux1_arch = SupportedArchitecture(
    name="FluxPipeline",
    task=PipelineTask.PIXEL_GENERATION,
    default_encoding="bfloat16",
    supported_encodings={"bfloat16"},
    example_repo_ids=[
        "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.1-schnell",
    ],
    pipeline_model=FluxPipeline,  # type: ignore[arg-type]
    context_type=PixelContext,
    config=FluxArchConfig,
    default_weights_format=WeightsFormat.safetensors,
    tokenizer=PixelGenerationTokenizer,
)
