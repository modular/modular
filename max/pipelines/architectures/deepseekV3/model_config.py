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
"""Config for DeepseekV3 models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams, KVCacheStrategy
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, MAXModelConfigBase
from transformers import AutoConfig


@dataclass
class DeepseekV3ConfigBase(MAXModelConfigBase):
    """Base configuration for DeepseekV3 models."""

    # MAX specific fields
    dtype: DType
    kv_params: KVCacheParams
    devices: list[DeviceRef]

    vocab_size: int = 129280
    hidden_size: int = 7168
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    moe_layer_freq: int = 1
    num_hidden_layers: int = 61
    num_attention_heads: int = 128
    num_key_value_heads: int = 128
    n_shared_experts: int = 1
    n_routed_experts: int = 256
    routed_scaling_factor: float = 2.5
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    topk_method: str = "greedy"
    n_group: int = 8
    topk_group: int = 4
    num_experts_per_tok: int = 8
    first_k_dense_replace: int = 3
    norm_topk_prob: bool = True
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: dict[str, Any] | None = None
    rope_interleave: bool = True
    scoring_func: str = "sigmoid"
    attention_bias: bool = False
    attention_dropout: float = 0.0

    graph_mode: str = "auto"  # "auto" | "prefill" | "decode"

    def __post_init__(self):
        if self.hidden_act != "silu":
            raise ValueError(
                "'silu' is the only hidden_act currently supported"
            )

        if self.rope_scaling and self.rope_scaling["type"] != "yarn":
            raise ValueError(
                "'yarn' is the only rope_scaling type currently supported"
            )

        if self.tie_word_embeddings:
            raise ValueError("tie_word_embeddings is not supported yet")

    @staticmethod
    def help() -> dict[str, str]:
        return {}


@dataclass
class DeepseekV3Config(MAXModelConfig, DeepseekV3ConfigBase):
    @staticmethod
    def help() -> dict[str, str]:
        return {}

    @staticmethod
    def get_kv_params(
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
        page_size: int = 128,
        data_parallel_degree: int = 1,
    ) -> KVCacheParams:
        return KVCacheParams(
            dtype=cache_dtype,
            # n_kv_heads should always be 1 because we only cache a single latent vector
            # in LatentAttention
            n_kv_heads=1,
            head_dim=huggingface_config.kv_lora_rank
            + huggingface_config.qk_rope_head_dim,
            cache_strategy=KVCacheStrategy.PAGED,
            n_devices=n_devices,
            page_size=page_size,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
            enable_kvcache_swapping_to_host=kv_cache_config.enable_kvcache_swapping_to_host,
            host_kvcache_swap_space_gb=kv_cache_config.host_kvcache_swap_space_gb,
            data_parallel_degree=data_parallel_degree,
        )
