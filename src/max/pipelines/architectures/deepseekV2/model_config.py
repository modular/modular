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
"""Config for DeepseekV2 models."""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams
from max.pipelines.max_config import (
    KVCacheConfig,
    MAXModelConfig,
    MAXModelConfigBase,
)
from transformers import AutoConfig


@dataclass
class DeepseekV2ConfigBase(MAXModelConfigBase):
    """Base configuration for DeepseekV2 models."""

    # MAX specific fields
    dtype: DType
    kv_params: KVCacheParams
    devices: list[DeviceRef]

    # Default values lifted from Transformers DeepseekV2Config
    vocab_size: int = 102400
    hidden_size: int = 4096
    intermediate_size: int = 11008
    moe_intermediate_size: int = 1407
    num_hidden_layers: int = 30
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    n_shared_experts: int = 0
    n_routed_experts: int = 0
    ep_size: int = 1
    routed_scaling_factor: float = 1.0
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    topk_method: str = "greedy"
    n_group: int = 0
    topk_group: int = 0
    num_experts_per_tok: int = 0
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 0
    norm_topk_prob: bool = False
    scoring_func: str = "softmax"
    aux_loss_alpha: float = 0.001
    seq_aux: bool = True
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int = 100000
    eos_token_id: int = 100001
    pretraining_tp: int = 1
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: dict | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0

    def __post_init__(self):
        if self.topk_method != "greedy":
            raise ValueError(
                "'greedy' is the only topk_method currently supported"
            )

        if self.hidden_act != "silu":
            raise ValueError(
                "'silu' is the only hidden_act currently supported"
            )

        if self.rope_scaling and self.rope_scaling["type"] != "yarn":
            raise ValueError(
                "'yarn' is the only rope_scaling type currently supported"
            )

        if self.norm_topk_prob:
            raise ValueError("norm_topk_prob is not supported yet")

        if self.pretraining_tp != 1:
            raise ValueError("Training not supported at this time")

        if self.tie_word_embeddings:
            raise ValueError("tie_word_embeddings is not supported yet")

        if self.pad_token_id != None:
            raise ValueError("Padding token is not supported yet")

    @staticmethod
    def help() -> dict[str, str]:
        return {}


@dataclass
class DeepseekV2Config(MAXModelConfig, DeepseekV2ConfigBase):
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
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=huggingface_config.kv_lora_rank
            + huggingface_config.qk_rope_head_dim,
            cache_strategy=kv_cache_config.cache_strategy,
            page_size=kv_cache_config.kv_cache_page_size,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
            enable_kvcache_swapping_to_host=kv_cache_config.enable_kvcache_swapping_to_host,
            host_kvcache_swap_space_gb=kv_cache_config.host_kvcache_swap_space_gb,
            n_devices=n_devices,
        )
