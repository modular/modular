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
"""Config for DeepseekV3 NextN (Next-N token prediction) models."""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.legacy.kv_cache import KVCacheParams, KVCacheStrategy
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from transformers import AutoConfig

from ..deepseekV3.model_config import DeepseekV3Config


@dataclass(kw_only=True)
class DeepseekV3NextNConfig(DeepseekV3Config):
    """Configuration for DeepseekV3 NextN model.

    The NextN (Next-N token prediction) model is a single-layer decoder that takes
    both input embeddings and hidden states from a base model as input, concatenates
    them, and processes through a single decoder layer to predict the next token.
    """

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
        page_size: int = 128,
    ) -> KVCacheParams:
        """Get KV cache parameters for the NextN model.

        The NextN model has only a single decoder layer, so we only need to cache
        one layer's worth of KV pairs.
        """
        data_parallel_degree = pipeline_config.model.data_parallel_degree
        if len(devices) != data_parallel_degree:
            raise ValueError(
                "Number of devices must match data parallel degree"
            )
        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=1,
            head_dim=huggingface_config.kv_lora_rank
            + huggingface_config.qk_rope_head_dim,
            num_layers=1,  # MTP only has a single decoder layer
            cache_strategy=KVCacheStrategy.PAGED,
            devices=devices,
            page_size=page_size,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
            enable_kvcache_swapping_to_host=kv_cache_config.enable_kvcache_swapping_to_host,
            host_kvcache_swap_space_gb=kv_cache_config.host_kvcache_swap_space_gb,
            data_parallel_degree=data_parallel_degree,
            is_mla=True,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        """NextN only has a single decoder layer."""
        return 1
