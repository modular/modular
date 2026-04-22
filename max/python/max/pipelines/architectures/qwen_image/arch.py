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
from max.pipelines.core import PixelContext
from max.pipelines.lib import (
    PixelGenerationTokenizer,
    SupportedArchitecture,
)
from max.pipelines.lib.config import MAXModelConfig, PipelineConfig
from max.pipelines.lib.interfaces import ArchConfig
from typing_extensions import Self

from .pipeline_qwen_image import QwenImagePipeline


@dataclass(kw_only=True)
class QwenImageArchConfig(ArchConfig):
    """Pipeline-level config for QwenImage (implements ArchConfig; no KV cache)."""

    pipeline_config: PipelineConfig

    def get_max_seq_len(self) -> int:
        return 0  # Not used for pixel generation.

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
    ) -> Self:
        if len(pipeline_config.model.device_specs) != 1:
            raise ValueError("QwenImage is only supported on a single device")
        return cls(pipeline_config=pipeline_config)


qwen_image_arch = SupportedArchitecture(
    name="QwenImagePipeline",
    task=PipelineTask.PIXEL_GENERATION,
    default_encoding="bfloat16",
    supported_encodings={"bfloat16"},
    example_repo_ids=[
        "Qwen/Qwen-Image-2512",
    ],
    pipeline_model=QwenImagePipeline,  # type: ignore[arg-type]
    context_type=PixelContext,
    default_weights_format=WeightsFormat.safetensors,
    tokenizer=PixelGenerationTokenizer,
    config=QwenImageArchConfig,
)
