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
from max.interfaces import PipelineTask
from max.pipelines.core import validate_wan_max_pixel_area
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.lib.config import MAXModelConfig, PipelineConfig
from max.pipelines.lib.interfaces import ArchConfig
from typing_extensions import Self

from .context import WanContext
from .tokenizer import WanTokenizer
from .wan_executor import WanExecutor


@dataclass(kw_only=True)
class WanArchConfig(ArchConfig):
    """Pipeline-level config for Wan (implements ArchConfig; no KV cache)."""

    pipeline_config: PipelineConfig

    def get_max_seq_len(self) -> int:
        # Tokenizer padding length — matches diffusers __call__ default.
        return 512

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
    ) -> Self:
        model_config = model_config or pipeline_config.model
        if len(model_config.device_specs) != 1:
            raise ValueError("Wan is only supported on a single device")
        return cls(pipeline_config=pipeline_config)


wan_arch = SupportedArchitecture(
    name="WanPipeline",
    task=PipelineTask.PIXEL_GENERATION,
    default_encoding="bfloat16",
    supported_encodings={"bfloat16", "float32"},
    example_repo_ids=[
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "yetter-ai/Wan2.2-TI2V-5B-Turbo-Diffusers",
    ],
    pipeline_model=WanExecutor,
    context_type=WanContext,
    default_weights_format=WeightsFormat.safetensors,
    tokenizer=WanTokenizer,
    config=WanArchConfig,
    context_validators=[validate_wan_max_pixel_area],
)

wan_i2v_arch = SupportedArchitecture(
    name="WanImageToVideoPipeline",
    task=PipelineTask.PIXEL_GENERATION,
    default_encoding="bfloat16",
    supported_encodings={"bfloat16", "float32"},
    example_repo_ids=[
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
    ],
    pipeline_model=WanExecutor,
    context_type=WanContext,
    default_weights_format=WeightsFormat.safetensors,
    tokenizer=WanTokenizer,
    config=WanArchConfig,
    context_validators=[validate_wan_max_pixel_area],
)
