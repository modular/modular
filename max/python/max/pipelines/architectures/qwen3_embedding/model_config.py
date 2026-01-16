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
"""Config for Qwen3 Embedding models."""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams
from max.pipelines.architectures.qwen3.model_config import Qwen3Config
from max.pipelines.lib import KVCacheConfig, MAXModelConfigBase, PipelineConfig
from transformers import AutoConfig


class Qwen3EmbeddingConfig(MAXModelConfigBase):
    """Configuration for Qwen3 Embedding models.

    Reuses Qwen3Config for most parameters but adapts KV cache
    configuration for embedding task.
    """

    @staticmethod
    def help() -> dict[str, str]:
        return {}

    @staticmethod
    def get_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Get KV cache parameters for Qwen3 Embedding.

        Delegates to Qwen3Config since the embedding model uses the same
        architecture.
        """
        return Qwen3Config.get_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        """Get the number of transformer layers."""
        return Qwen3Config.get_num_layers(huggingface_config)
