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
"""Config for Qwen3 models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from transformers.models.auto.configuration_auto import AutoConfig
from typing_extensions import Self, override

from ..llama3.model_config import Llama3Config


@dataclass(kw_only=True)
class Qwen3Config(Llama3Config):
    # MoE parameters - these are optional and only used for Qwen3-MOE models
    num_experts: int = 0
    """Number of experts in the MoE layer. 0 means dense model (no MoE)."""

    num_experts_per_tok: int = 1
    """Number of experts per token in the MoE layer."""

    moe_intermediate_size: int = 0
    """Intermediate size in the MoE layer. If 0, uses intermediate_size."""

    mlp_only_layers: list[int] = field(default_factory=list)
    """List of layer indices that use MLP instead of MoE."""

    norm_topk_prob: bool = False
    """Whether to use top-k probability normalization in the MoE layer."""

    decoder_sparse_step: int = 1
    """Sparse step for the decoder. Controls which layers use MoE."""

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Override the default Llama3Config.construct_kv_params to use head_dim from config.

        Qwen3 models have an explicit head_dim field in their configuration,
        unlike Llama models where it needs to be calculated.

        Args:
            huggingface_config: The HuggingFace configuration object.
            pipeline_config: The MAX Engine pipeline configuration.
            devices: Devices to use for the KV cache.
            kv_cache_config: Configuration for KV cache.
            cache_dtype: Data type for the cache.

        Returns:
            KVCacheParams object with the correct head_dim from config.
        """
        data_parallel_degree = pipeline_config.model.data_parallel_degree
        if data_parallel_degree > 1:
            raise ValueError(
                "Data parallelism is not supported for Qwen3 models"
            )
        return kv_cache_config.to_params(
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=huggingface_config.head_dim,
            num_layers=Qwen3Config.get_num_layers(huggingface_config),
            devices=devices,
            data_parallel_degree=data_parallel_degree,
        )

    @staticmethod
    def calculate_attention_multiplier(huggingface_config: AutoConfig) -> float:
        """The attention multiplier for Qwen3 models.

        Uses the explicit head_dim from the config instead of calculating it.

        Args:
            huggingface_config: The HuggingFace configuration object.

        Returns:
            The attention multiplier value.
        """
        return getattr(
            huggingface_config,
            "attention_multiplier",
            math.sqrt(1.0 / float(huggingface_config.head_dim)),
        )

    @override
    @classmethod
    def initialize(cls, pipeline_config: PipelineConfig) -> Self:
        """Initializes a Qwen3Config instance from pipeline configuration.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.

        Returns:
            An initialized Qwen3Config instance.
        """
        huggingface_config = pipeline_config.model.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for '{pipeline_config.model.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )
        return cls.initialize_from_config(pipeline_config, huggingface_config)

    @override
    @classmethod
    def initialize_from_config(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> Self:
        """Initializes a Qwen3Config instance from pipeline and HuggingFace configs.

        This method creates a config instance with all fields that can be determined
        from the pipeline configuration, without needing the state_dict.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            huggingface_config: The HuggingFace model configuration.

        Returns:
            An initialized Qwen3Config instance.
        """
        # Get base config from Llama3Config
        base_config = Llama3Config.initialize_from_config(
            pipeline_config, huggingface_config
        )

        kv_cache_config = pipeline_config.model.kv_cache
        quantization_encoding = pipeline_config.model.quantization_encoding
        if quantization_encoding is None:
            raise ValueError("quantization_encoding must not be None")
        cache_dtype = pipeline_config.model.kv_cache.cache_dtype
        n_devices = len(pipeline_config.model.device_specs)

        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model.device_specs[:n_devices]
        ]

        # Override the KV parameters and attention multiplier with Qwen3-specific calculations
        qwen3_kv_params = Qwen3Config.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

        qwen3_attention_multiplier = Qwen3Config.calculate_attention_multiplier(
            huggingface_config=huggingface_config,
        )

        # Handle both MoE (e.g., Qwen3-30B-A3B) and dense (e.g., Qwen3-8B) variants.
        # For dense models, num_experts=0 ensures the decoder always uses MLP layers
        num_experts = getattr(huggingface_config, "num_experts", 0)
        num_experts_per_tok = getattr(
            huggingface_config, "num_experts_per_tok", 1
        )
        moe_intermediate_size = getattr(
            huggingface_config,
            "moe_intermediate_size",
            base_config.intermediate_size,
        )
        mlp_only_layers = getattr(huggingface_config, "mlp_only_layers", [])
        norm_topk_prob = getattr(huggingface_config, "norm_topk_prob", False)
        decoder_sparse_step = getattr(
            huggingface_config, "decoder_sparse_step", 1
        )

        # Return a new Qwen3Config with the corrected parameters
        return cls(
            hidden_size=base_config.hidden_size,
            num_attention_heads=base_config.num_attention_heads,
            num_key_value_heads=base_config.num_key_value_heads,
            num_hidden_layers=base_config.num_hidden_layers,
            rope_theta=base_config.rope_theta,
            rope_scaling_params=base_config.rope_scaling_params,
            rms_norm_eps=base_config.rms_norm_eps,
            intermediate_size=base_config.intermediate_size,
            interleaved_rope_weights=base_config.interleaved_rope_weights,
            vocab_size=base_config.vocab_size,
            dtype=base_config.dtype,
            model_quantization_encoding=base_config.model_quantization_encoding,
            quantization_config=base_config.quantization_config,
            max_seq_len=base_config.max_seq_len,
            kv_params=qwen3_kv_params,  # Use Qwen3-specific KV params
            attention_multiplier=qwen3_attention_multiplier,  # Use Qwen3-specific attention multiplier
            embedding_multiplier=base_config.embedding_multiplier,
            residual_multiplier=base_config.residual_multiplier,
            devices=base_config.devices,
            clip_qkv=base_config.clip_qkv,
            use_subgraphs=base_config.use_subgraphs,
            # MoE parameters
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            mlp_only_layers=mlp_only_layers,
            norm_topk_prob=norm_topk_prob,
            decoder_sparse_step=decoder_sparse_step,
        )
