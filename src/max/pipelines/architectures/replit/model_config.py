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
"""Config for Replit models."""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef
from max.nn import ReturnLogits
from max.nn.kv_cache import KVCacheParams
from max.pipelines.max_config import KVCacheConfig
from max.pipelines.model_config import MAXModelConfig, MAXModelConfigBase
from transformers import AutoConfig


@dataclass
class ReplitConfigBase(MAXModelConfigBase):
    """Base configuration for Llama3 models."""

    # Required fields
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    vocab_size: int
    dtype: DType
    kv_params: KVCacheParams
    return_logits: ReturnLogits

    attention_multiplier: float
    devices: list[DeviceRef]

    @staticmethod
    def help() -> dict[str, str]:
        return {}


@dataclass
class ReplitConfig(MAXModelConfig, ReplitConfigBase):
    @staticmethod
    def help() -> dict[str, str]:
        return {}

    @staticmethod
    def get_kv_params(
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.attn_config["kv_n_heads"],
            head_dim=huggingface_config.d_model // huggingface_config.n_heads,
            cache_strategy=kv_cache_config.cache_strategy,
            page_size=kv_cache_config.kv_cache_page_size,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
            enable_kvcache_swapping_to_host=kv_cache_config.enable_kvcache_swapping_to_host,
            host_kvcache_swap_space_gb=kv_cache_config.host_kvcache_swap_space_gb,
            n_devices=n_devices,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        return huggingface_config.n_layers
