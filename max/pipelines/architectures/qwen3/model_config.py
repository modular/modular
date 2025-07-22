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
"""Config for Qwen3 models."""

from __future__ import annotations

import math
from typing import Callable, Literal

from max.dtype import DType
from max.graph import TensorValue
from max.graph.weights import WeightData
from max.nn import ReturnLogits
from max.nn.kv_cache import KVCacheParams
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from transformers.models.auto.configuration_auto import AutoConfig

from ..llama3.model_config import Llama3Config


class Qwen3Config(Llama3Config):
    """Implementation of MAXModelConfig for Qwen3 models.

    Qwen3 models use a different approach for head_dim calculation compared to Llama3.
    Llama3 calculates head_dim as hidden_size // num_attention_heads,
    Qwen3 models have an explicit head_dim field in their configuration.
    """

    @staticmethod
    def get_kv_params(
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
        pipeline_parallel_degree: int = 1,
    ) -> KVCacheParams:
        """Override the default Llama3Config.get_kv_params to use head_dim from config.

        Qwen3 models have an explicit head_dim field in their configuration,
        unlike Llama models where it needs to be calculated.

        Args:
            huggingface_config: The HuggingFace configuration object.
            n_devices: Number of devices for distributed inference.
            kv_cache_config: Configuration for KV cache.
            cache_dtype: Data type for the cache.
            pipeline_parallel_degree: Pipeline parallel degree (default 1).

        Returns:
            KVCacheParams object with the correct head_dim from config.
        """
        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=getattr(huggingface_config, "num_key_value_heads"),  # noqa: B009
            head_dim=getattr(  # noqa: B009
                huggingface_config, "head_dim"
            ),  # Use head_dim directly from config
            page_size=kv_cache_config.kv_cache_page_size,
            cache_strategy=kv_cache_config.cache_strategy,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
            enable_kvcache_swapping_to_host=kv_cache_config.enable_kvcache_swapping_to_host,
            host_kvcache_swap_space_gb=kv_cache_config.host_kvcache_swap_space_gb,
            n_devices=n_devices,
            # Pipeline parallel fields
            pipeline_parallel_degree=pipeline_parallel_degree,
            total_num_layers=huggingface_config.num_hidden_layers
            if pipeline_parallel_degree > 1
            else None,
        )

    @staticmethod
    def calculate_attention_multiplier(
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
        pipeline_parallel_degree: int = 1,
    ) -> float:
        """The attention multiplier for Qwen3 models.

        Uses the explicit head_dim from the config instead of calculating it.

        Args:
            huggingface_config: The HuggingFace configuration object.
            n_devices: Number of devices for distributed inference.
            kv_cache_config: Configuration for KV cache.
            cache_dtype: Data type for the cache.
            pipeline_parallel_degree: Pipeline parallel degree (default 1).

        Returns:
            The attention multiplier value.
        """
        return getattr(
            huggingface_config,
            "attention_multiplier",
            math.sqrt(
                1.0
                / float(
                    Qwen3Config.get_kv_params(
                        huggingface_config=huggingface_config,
                        n_devices=n_devices,
                        kv_cache_config=kv_cache_config,
                        cache_dtype=cache_dtype,
                        pipeline_parallel_degree=pipeline_parallel_degree,
                    ).head_dim
                )
            ),
        )

    @staticmethod
    def generate(
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        dtype: DType,
        n_devices: int,
        logits_postprocessor: Callable[[TensorValue], TensorValue] | None,
        cache_dtype: DType,
        kv_cache_config: KVCacheConfig,
        return_logits: ReturnLogits,
        norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm",
        attention_bias: bool = False,
        pipeline_parallel_degree: int = 1,
        tensor_parallel_degree: int = 1,
    ) -> Qwen3Config:
        """Generate a Qwen3Config from the provided parameters.

        This method largely delegates to Llama3Config.generate but ensures
        the correct attention_multiplier calculation using Qwen3's head_dim.
        - GGUF is not currently supported for Qwen3.

        Args:
            pipeline_config: Pipeline configuration.
            huggingface_config: HuggingFace model configuration.
            state_dict: Model state dictionary.
            dtype: Model data type.
            n_devices: Number of devices.
            logits_postprocessor: Optional logits postprocessor.
            cache_dtype: KV cache data type.
            kv_cache_config: KV cache configuration.
            return_logits: Return logits configuration.
            norm_method: Normalization method.
            attention_bias: Whether to use attention bias.
            pipeline_parallel_degree: Pipeline parallel degree.
            tensor_parallel_degree: Tensor parallel degree.

        Returns:
            Configured Qwen3Config instance.
        """
        # Call the parent generate method to get most of the configuration
        base_config = Llama3Config.generate(
            pipeline_config=pipeline_config,
            huggingface_config=huggingface_config,
            state_dict=state_dict,
            dtype=dtype,
            n_devices=n_devices,
            logits_postprocessor=logits_postprocessor,
            cache_dtype=cache_dtype,
            kv_cache_config=kv_cache_config,
            return_logits=return_logits,
            norm_method=norm_method,
            attention_bias=attention_bias,
            pipeline_parallel_degree=pipeline_parallel_degree,
            tensor_parallel_degree=tensor_parallel_degree,
        )

        # Override the KV parameters and attention multiplier with Qwen3-specific calculations
        qwen3_kv_params = Qwen3Config.get_kv_params(
            huggingface_config=huggingface_config,
            n_devices=n_devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
            pipeline_parallel_degree=pipeline_parallel_degree,
        )

        qwen3_attention_multiplier = Qwen3Config.calculate_attention_multiplier(
            huggingface_config=huggingface_config,
            n_devices=n_devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
            pipeline_parallel_degree=pipeline_parallel_degree,
        )

        # Return a new Qwen3Config with the corrected parameters
        return Qwen3Config(
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
            return_logits=base_config.return_logits,
            max_seq_len=base_config.max_seq_len,
            kv_params=qwen3_kv_params,  # Use Qwen3-specific KV params
            norm_method=base_config.norm_method,
            norm_dtype=base_config.norm_dtype,
            attention_bias=base_config.attention_bias,
            tie_word_embeddings=base_config.tie_word_embeddings,
            stacked_mlp=base_config.stacked_mlp,
            stacked_qkv=base_config.stacked_qkv,
            logits_postprocessor=base_config.logits_postprocessor,
            attention_multiplier=qwen3_attention_multiplier,  # Use Qwen3-specific attention multiplier
            embedding_multiplier=base_config.embedding_multiplier,
            residual_multiplier=base_config.residual_multiplier,
            devices=base_config.devices,
            clip_qkv=base_config.clip_qkv,
            float8_config=base_config.float8_config,
            use_subgraphs=base_config.use_subgraphs,
            pipeline_parallel_degree=pipeline_parallel_degree,
            tensor_parallel_degree=tensor_parallel_degree,
            dist_gemm_config=base_config.dist_gemm_config,
        )
