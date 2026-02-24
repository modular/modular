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
"""Config for DeepseekV3.2 models."""

from __future__ import annotations

from dataclasses import dataclass

from max.graph import DeviceRef
from max.pipelines.architectures.deepseekV3.model_config import DeepseekV3Config
from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.config.config_enums import supported_encoding_dtype
from typing_extensions import Self, override


@dataclass(kw_only=True)
class DeepseekV3_2Config(DeepseekV3Config):
    """Configuration for DeepseekV3.2 models."""

    # Added parameters for the Indexer used in DeepSeek Sparse Attention.
    index_head_dim: int = 128
    index_n_heads: int = 64
    index_topk: int = 2048

    @override
    @classmethod
    def initialize(cls, pipeline_config: PipelineConfig) -> Self:
        """Initializes a DeepseekV3_2Config instance from pipeline configuration.

        This method creates a config instance with all fields that can be determined
        from the pipeline configuration, without needing the state_dict.
        Fields that depend on the state_dict (like norm_dtype, float8_config, etc.)
        should be set directly after calling this method.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.

        Returns:
            An initialized DeepseekV3_2Config instance.
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

        return cls(
            dtype=dtype,
            kv_params=kv_params,
            devices=device_refs,
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
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=config.tie_word_embeddings,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            rope_interleave=getattr(config, "rope_interleave", True),
            scoring_func=config.scoring_func,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
            # DeepseekV3.2 specific fields
            index_head_dim=config.index_head_dim,
            index_n_heads=config.index_n_heads,
            index_topk=config.index_topk,
        )
