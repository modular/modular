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

    max_seq_len: int = 77
    secondary_max_seq_len: int = 512

    def get_max_seq_len(self) -> int:
        """Returns the maximum sequence length for the primary tokenizer."""
        return self.max_seq_len

    @classmethod
    def initialize(cls, pipeline_config: PipelineConfig) -> Self:
        if len(pipeline_config.model.device_specs) != 1:
            raise ValueError("Flux1 is only supported on a single device")
        return cls()


flux1_modulev3_arch = SupportedArchitecture(
    name="FluxPipeline_ModuleV3",
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
