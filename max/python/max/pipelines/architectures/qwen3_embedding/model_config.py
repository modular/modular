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

from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.interfaces.arch_config import ArchConfig
from typing_extensions import Self


@dataclass(kw_only=True)
class Qwen3EmbeddingConfig(ArchConfig):
    """Qwen3 embedding model configuration."""

    pipeline_config: PipelineConfig

    def get_max_seq_len(self) -> int:
        # Use configured max_length, bounded by model's max_position_embeddings
        huggingface_config = self.pipeline_config.model.huggingface_config
        model_max = getattr(
            huggingface_config, "max_position_embeddings", 32768
        )
        configured_max = self.pipeline_config.model.max_length or 8192

        if configured_max > model_max:
            raise ValueError(
                f"Configured max_length ({configured_max}) exceeds model's "
                f"max_position_embeddings ({model_max})"
            )

        return configured_max

    @classmethod
    def initialize(cls, pipeline_config: PipelineConfig) -> Self:
        return cls(pipeline_config=pipeline_config)
