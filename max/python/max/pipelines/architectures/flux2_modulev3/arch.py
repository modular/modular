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
from max.pipelines.lib import PixelGenerationTokenizer, SupportedArchitecture
from max.pipelines.lib.config import MAXModelConfig, PipelineConfig
from max.pipelines.lib.interfaces import ArchConfig
from typing_extensions import Self

from .pipeline_flux2 import Flux2Pipeline
from .pipeline_flux2_klein import Flux2KleinPipeline

_V2_NOT_IMPLEMENTED_MSG = (
    "ModuleV2 pipeline not yet implemented for this FLUX variant. "
    "Please add --prefer-module-v3 to the command to run the v3 variant."
)


class _Flux2KleinV2NotImplemented:
    """Stub v2 pipeline for FLUX.2-Klein — raises until a real v2 is implemented."""

    not_implemented_message: str = _V2_NOT_IMPLEMENTED_MSG

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(_V2_NOT_IMPLEMENTED_MSG)


@dataclass(kw_only=True)
class Flux2ArchConfig(ArchConfig):
    """Pipeline-level config for Flux2 (implements ArchConfig; no KV cache)."""

    max_seq_len: int = 512

    def get_max_seq_len(self) -> int:
        """Returns the maximum sequence length for the tokenizer."""
        return self.max_seq_len

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
    ) -> Self:
        model_config = model_config or pipeline_config.models["transformer"]
        if len(model_config.device_specs) != 1:
            raise ValueError("Flux2 is only supported on a single device")
        return cls()


flux2_modulev3_arch = SupportedArchitecture(
    name="Flux2Pipeline_ModuleV3",
    task=PipelineTask.PIXEL_GENERATION,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    default_encoding="bfloat16",
    supported_encodings={"bfloat16", "float4_e2m1fnx2"},
    example_repo_ids=[
        "black-forest-labs/FLUX.2-dev",
        "black-forest-labs/FLUX.2-dev-NVFP4",
    ],
    pipeline_model=Flux2Pipeline,  # type: ignore[arg-type]
    context_type=PixelContext,
    default_weights_format=WeightsFormat.safetensors,
    tokenizer=PixelGenerationTokenizer,
    config=Flux2ArchConfig,
)

flux2_klein_arch = SupportedArchitecture(
    name="Flux2KleinPipeline",
    task=PipelineTask.PIXEL_GENERATION,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    default_encoding="bfloat16",
    supported_encodings={"bfloat16"},
    example_repo_ids=[
        "black-forest-labs/FLUX.2-klein-4B",
        "black-forest-labs/FLUX.2-klein-9B",
        "black-forest-labs/FLUX.2-klein-base-4B",
        "black-forest-labs/FLUX.2-klein-base-9B",
    ],
    pipeline_model=_Flux2KleinV2NotImplemented,  # type: ignore[arg-type]
    context_type=PixelContext,
    default_weights_format=WeightsFormat.safetensors,
    tokenizer=PixelGenerationTokenizer,
    config=Flux2ArchConfig,
)

flux2_klein_modulev3_arch = SupportedArchitecture(
    name="Flux2KleinPipeline_ModuleV3",
    task=PipelineTask.PIXEL_GENERATION,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    default_encoding="bfloat16",
    supported_encodings={"bfloat16"},
    example_repo_ids=[
        "black-forest-labs/FLUX.2-klein-4B",
        "black-forest-labs/FLUX.2-klein-9B",
        "black-forest-labs/FLUX.2-klein-base-4B",
        "black-forest-labs/FLUX.2-klein-base-9B",
    ],
    pipeline_model=Flux2KleinPipeline,  # type: ignore[arg-type]
    context_type=PixelContext,
    default_weights_format=WeightsFormat.safetensors,
    tokenizer=PixelGenerationTokenizer,
    config=Flux2ArchConfig,
)
