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
"""Config for DeepseekV3 models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.comm.ep import EPConfig
from max.nn.float8_config import Float8Config
from max.nn.kv_cache import KVCacheParams, KVCacheQuantizationConfig
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from max.pipelines.lib.config_enums import supported_encoding_dtype
from max.pipelines.lib.interfaces.arch_config import ArchConfigWithKVCache
from max.pipelines.lib.utils import upper_bounded_default
from transformers import AutoConfig
from typing_extensions import Self, override


@dataclass(kw_only=True)
class DeepseekV3Config(ArchConfigWithKVCache):
    """Configuration for DeepseekV3 models."""

    # MAX specific fields
    dtype: DType
    kv_params: KVCacheParams
    devices: list[DeviceRef]
    use_subgraphs: bool = True
    data_parallel_degree: int = 1

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
    """Maximum positional embeddings as defined by the original model."""
    max_seq_len: int = 163840
    """Maximum sequence length as defined by the MAX Engine pipeline configuration."""

    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: dict[str, Any] | None = None
    rope_interleave: bool = True
    scoring_func: str = "sigmoid"
    attention_bias: bool = False
    attention_dropout: float = 0.0

    norm_dtype: DType = DType.bfloat16
    correction_bias_dtype: DType | None = None
    max_batch_context_length: int = 131072
    float8_config: Float8Config | None = None
    ep_config: EPConfig | None = None
    graph_mode: str = "auto"  # "auto" | "prefill" | "decode"

    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    """Whether to return the last token, all logits, or a variable number of logits."""

    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE
    """Whether to return hidden states and which type (none, last, all, last_normalized, all_normalized)."""

    def __post_init__(self) -> None:
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

        # Validate data parallel degree for DeepSeekV3.
        # DP attention requires DP degree to match device count.
        # TP attention requires DP degree to be 1.
        num_devices = len(self.devices)
        if self.data_parallel_degree not in (1, num_devices):
            raise ValueError(
                f"data_parallel_degree for DeepSeekV3 ({self.data_parallel_degree}) must be "
                f"1 (TP attention) or equal to the number of devices ({num_devices}). "
                "DP attention requires DP degree to match device count."
            )

    def get_kv_params(self) -> KVCacheParams:
        return self.kv_params

    def get_max_seq_len(self) -> int:
        return self.max_seq_len

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        data_parallel_degree = pipeline_config.model.data_parallel_degree

        kvcache_quant_config = None
        if kv_cache_config.cache_dtype in (
            DType.float8_e4m3fn,
            DType.float8_e4m3fnuz,
        ):
            # Configure the KVCacheParams quantization parameters.
            kvcache_quant_config = KVCacheQuantizationConfig(
                scale_dtype=DType.float32, quantization_granularity=32
            )
        return kv_cache_config.to_params(
            dtype=cache_dtype,
            # n_kv_heads should always be 1 because we only cache a single latent vector
            # in LatentAttention
            n_kv_heads=1,
            head_dim=huggingface_config.kv_lora_rank
            + huggingface_config.qk_rope_head_dim,
            num_layers=DeepseekV3Config.get_num_layers(huggingface_config),
            devices=devices,
            data_parallel_degree=data_parallel_degree,
            is_mla=True,
            kvcache_quant_config=kvcache_quant_config,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        return huggingface_config.num_hidden_layers

    @override
    @classmethod
    def initialize(cls, pipeline_config: PipelineConfig) -> Self:
        """Initializes a DeepseekV3Config instance from pipeline configuration.

        This method creates a config instance with all fields that can be determined
        from the pipeline configuration, without needing the state_dict.
        Fields that depend on the state_dict (like norm_dtype, float8_config, etc.)
        should be set via the `finalize()` method.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.

        Returns:
            An initialized DeepseekV3Config instance.
        """
        config = pipeline_config.model.huggingface_config
        if config is None:
            raise ValueError(
                f"HuggingFace config is required for '{pipeline_config.model.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )
        kv_cache_config = pipeline_config.model.kv_cache
        quantization_encoding = pipeline_config.model.quantization_encoding
        if quantization_encoding is None:
            raise ValueError("quantization_encoding must not be None")
        dtype = supported_encoding_dtype(quantization_encoding)
        cache_dtype = pipeline_config.model.kv_cache.cache_dtype

        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model.device_specs
        ]

        kv_params = cls.construct_kv_params(
            huggingface_config=config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

        max_seq_len = upper_bounded_default(
            upper_bound=config.max_position_embeddings,
            default=pipeline_config.model.max_length,
        )

        return cls(
            dtype=dtype,
            kv_params=kv_params,
            devices=device_refs,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            use_subgraphs=pipeline_config.model.use_subgraphs,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            moe_intermediate_size=config.moe_intermediate_size,
            moe_layer_freq=config.moe_layer_freq,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            n_shared_experts=config.n_shared_experts,
            n_routed_experts=config.n_routed_experts,
            routed_scaling_factor=config.routed_scaling_factor,
            kv_lora_rank=config.kv_lora_rank,
            q_lora_rank=config.q_lora_rank,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            qk_nope_head_dim=config.qk_nope_head_dim,
            topk_method=config.topk_method,
            n_group=config.n_group,
            topk_group=config.topk_group,
            num_experts_per_tok=config.num_experts_per_tok,
            first_k_dense_replace=config.first_k_dense_replace,
            norm_topk_prob=config.norm_topk_prob,
            hidden_act=config.hidden_act,
            max_position_embeddings=config.max_position_embeddings,
            max_seq_len=max_seq_len,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=config.tie_word_embeddings,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            rope_interleave=getattr(config, "rope_interleave", True),
            scoring_func=config.scoring_func,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
        )
