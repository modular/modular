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
"""Config for Pixtral models."""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef
from max.nn import ReturnLogits
from max.nn.kv_cache import KVCacheParams
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, MAXModelConfigBase
from transformers import AutoConfig


@dataclass
class PixtralConfigBase(MAXModelConfigBase):
    """Base configuration for Pixtral models."""

    # TODO: check if we need to add these fields
    dtype: DType
    devices: list[DeviceRef]

    # Llava fields
    image_token_index: int

    # Language model fields
    hidden_size: int
    num_attention_heads: int
    rms_norm_eps: float
    rope_theta: float
    max_seq_len: int
    num_hidden_layers: int
    head_dim: int
    num_key_value_heads: int
    feed_forward_length: int
    vocab_size: int
    kv_params: KVCacheParams
    return_logits: ReturnLogits
    attention_multiplier: float

    # Vision encoder fields
    patch_size: int
    image_size: int
    num_channels: int
    vision_hidden_size: int
    vision_num_attention_heads: int
    vision_rope_theta: float
    vision_num_hidden_layers: int
    vision_intermediate_size: int
    vision_head_dim: int


@dataclass
class PixtralConfig(MAXModelConfig, PixtralConfigBase):
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
            n_kv_heads=huggingface_config.text_config.num_key_value_heads,
            head_dim=huggingface_config.text_config.head_dim,
            cache_strategy=kv_cache_config.cache_strategy,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
            enable_kvcache_swapping_to_host=kv_cache_config.enable_kvcache_swapping_to_host,
            host_kvcache_swap_space_gb=kv_cache_config.host_kvcache_swap_space_gb,
            n_devices=n_devices,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        return huggingface_config.text_config.num_hidden_layers
