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

from __future__ import annotations

from dataclasses import dataclass

from max.graph.weights import WeightsFormat
from max.interfaces import InputModality, PipelineTask
from max.pipelines.core import PixelContext
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.lib.config import MAXModelConfig, PipelineConfig
from max.pipelines.lib.interfaces import ArchConfig
from typing_extensions import Self

from .flux2_executor import Flux2Executor
from .flux2_klein_executor import Flux2KleinExecutor
from .tokenizer import Flux2Tokenizer


@dataclass(kw_only=True)
class Flux2ArchConfig(ArchConfig):
    """Pipeline-level config for Flux2 (implements ArchConfig; no KV cache)."""

    pipeline_config: PipelineConfig

    def get_max_seq_len(self) -> int:
        return 512

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
    ) -> Self:
        return cls(pipeline_config=pipeline_config)


flux2_arch = SupportedArchitecture(
    name="Flux2Pipeline",
    task=PipelineTask.PIXEL_GENERATION,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    default_encoding="bfloat16",
    supported_encodings={"bfloat16", "float4_e2m1fnx2"},
    example_repo_ids=[
        "black-forest-labs/FLUX.2-dev",
        "black-forest-labs/FLUX.2-dev-NVFP4",
    ],
    pipeline_model=Flux2Executor,
    context_type=PixelContext,
    default_weights_format=WeightsFormat.safetensors,
    tokenizer=Flux2Tokenizer,
    config=Flux2ArchConfig,
)

flux2_klein_arch = SupportedArchitecture(
    name="Flux2KleinPipeline",
    task=PipelineTask.PIXEL_GENERATION,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    default_encoding="bfloat16",
    supported_encodings={"bfloat16", "float4_e2m1fnx2"},
    example_repo_ids=[
        "black-forest-labs/FLUX.2-klein-4B",
        "black-forest-labs/FLUX.2-klein-9B",
        "black-forest-labs/FLUX.2-klein-base-4B",
        "black-forest-labs/FLUX.2-klein-base-9B",
        "black-forest-labs/FLUX.2-klein-9b-nvfp4",
    ],
    pipeline_model=Flux2KleinExecutor,
    context_type=PixelContext,
    default_weights_format=WeightsFormat.safetensors,
    tokenizer=Flux2Tokenizer,
    config=Flux2ArchConfig,
)
