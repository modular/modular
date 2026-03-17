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
"""Config for MPNet V3 models."""

from __future__ import annotations

from dataclasses import dataclass

from max.pipelines.lib import MAXModelConfig, PipelineConfig
from max.pipelines.lib.interfaces.arch_config import ArchConfig
from max.pipelines.lib.utils import upper_bounded_default
from transformers import AutoConfig
from typing_extensions import Self, override


@dataclass(kw_only=True)
class MPNetConfig(ArchConfig):
    """Configuration for MPNet V3 models."""

    pool_embeddings: bool
    huggingface_config: AutoConfig
    pipeline_config: PipelineConfig

    def get_max_seq_len(self) -> int:
        try:
            return upper_bounded_default(
                upper_bound=self.huggingface_config.max_position_embeddings,
                default=self.pipeline_config.model.max_length,
            )
        except ValueError as e:
            raise ValueError(
                "Unable to infer max_length for MPNet, the provided "
                f"max_length ({self.pipeline_config.model.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({self.huggingface_config.max_position_embeddings})."
            ) from e

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
    ) -> Self:
        model_config = model_config or pipeline_config.model
        if len(model_config.device_specs) != 1:
            raise ValueError("MPNet model is only supported on a single device")
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for '{model_config.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )
        return cls(
            pool_embeddings=model_config.pool_embeddings,
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
        )
