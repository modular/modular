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
"""Config for Mistral models."""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.kv_cache import KVCacheParams
from max.pipelines.max_config import (
    KVCacheConfig,
    MAXModelConfig,
    MAXModelConfigBase,
)
from transformers import AutoConfig


@dataclass
class MistralConfigBase(MAXModelConfigBase):
    """Base configuration for Llama3 models."""

    # Required fields
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    head_dim: int
    vocab_size: int
    rope_theta: float
    max_seq_len: int
    rms_norm_eps: float
    feed_forward_length: int

    dtype: DType
    kv_params: KVCacheParams
    return_n_logits: int

    attention_multiplier: float
    devices: list[DeviceRef]

    @staticmethod
    def help() -> dict[str, str]:
        return {}


@dataclass
class MistralConfig(MAXModelConfig, MistralConfigBase):
    @staticmethod
    def help() -> dict[str, str]:
        return {}

    # TODO(zheng): Figure out a scalable abstract method for all MAXModelConfigs.
    @staticmethod
    def get_kv_params(
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return KVCacheParams(
            page_size=kv_cache_config.kv_cache_page_size,
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=huggingface_config.head_dim,
            cache_strategy=kv_cache_config.cache_strategy,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
            enable_kvcache_swapping_to_host=kv_cache_config.enable_kvcache_swapping_to_host,
            host_kvcache_swap_space_gb=kv_cache_config.host_kvcache_swap_space_gb,
            n_devices=n_devices,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        return huggingface_config.num_hidden_layers
