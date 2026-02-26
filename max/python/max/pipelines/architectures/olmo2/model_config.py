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
"""Config for Olmo2 models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from transformers.models.auto.configuration_auto import AutoConfig
from typing_extensions import Self, override

from ..llama3.model_config import Llama3Config


@dataclass(kw_only=True)
class Olmo2Config(Llama3Config):
    """Implementation of MAXModelConfig for Olmo2 models.
    Olmo2 models use a different approach for head_dim calculation compared to Llama3.
    Llama3 calculates head_dim as hidden_size // num_attention_heads,
    Olmo2 models have an explicit head_dim field in their configuration.
    """

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Override the default Llama3Config.construct_kv_params to use head_dim from config.
        Olmo2 models have an explicit head_dim field in their configuration,
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
                "Data parallelism is not supported for Olmo2 models"
            )
        return kv_cache_config.to_params(
            dtype=cache_dtype,
            n_kv_heads=getattr(huggingface_config, "num_key_value_heads"),  # noqa: B009
            head_dim=Olmo2Config.get_head_dim(huggingface_config),
            num_layers=Olmo2Config.get_num_layers(huggingface_config),
            devices=devices,
            data_parallel_degree=data_parallel_degree,
        )

    @staticmethod
    def get_head_dim(huggingface_config: AutoConfig) -> int:
        if hasattr(huggingface_config, "head_dim"):
            return huggingface_config.head_dim
        else:
            return (
                huggingface_config.hidden_size
                // huggingface_config.num_attention_heads
            )

    @staticmethod
    def calculate_attention_multiplier(huggingface_config: AutoConfig) -> float:
        """The attention multiplier for Olmo2 models.
        Uses the explicit head_dim from the config instead of calculating it.
        Args:
            huggingface_config: The HuggingFace configuration object.
        Returns:
            The attention multiplier value.
        """
        return getattr(
            huggingface_config,
            "attention_multiplier",
            math.sqrt(
                1.0 / float(Olmo2Config.get_head_dim(huggingface_config))
            ),
        )

    @override
    @classmethod
    def initialize(cls, pipeline_config: PipelineConfig) -> Self:
        huggingface_config = pipeline_config.model.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for '{pipeline_config.model.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )
        return cls.initialize_from_config(pipeline_config, huggingface_config)

    @classmethod
    def initialize_from_config(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> Self:
        """Initializes an Olmo2Config instance from pipeline and HuggingFace configuration.

        This method creates a config instance with all fields that can be determined
        from the pipeline and HuggingFace configuration, without needing the state_dict.
        Fields that depend on the state_dict (like tie_word_embeddings, float8_config)
        should be set via the `finalize()` method.

        Overrides Llama3Config.initialize_from_config to use Olmo2-specific
        KV params and attention multiplier calculations.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            huggingface_config: The HuggingFace model configuration object.

        Returns:
            An initialized Olmo2Config instance.
        """
        # Get the base config from Llama3Config
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

        # Override the KV parameters and attention multiplier with Olmo2-specific calculations
        olmo2_kv_params = Olmo2Config.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

        olmo2_attention_multiplier = Olmo2Config.calculate_attention_multiplier(
            huggingface_config=huggingface_config,
        )

        # Return a new Olmo2Config with the corrected parameters
        return cls(
            hidden_size=base_config.hidden_size,
            num_attention_heads=base_config.num_attention_heads,
            num_key_value_heads=base_config.num_key_value_heads,
            num_hidden_layers=base_config.num_hidden_layers,
            rope_theta=base_config.rope_theta,
            rope_scaling_params=base_config.rope_scaling_params,
            longrope_scaling_params=base_config.longrope_scaling_params,
            rms_norm_eps=base_config.rms_norm_eps,
            intermediate_size=base_config.intermediate_size,
            interleaved_rope_weights=base_config.interleaved_rope_weights,
            vocab_size=base_config.vocab_size,
            dtype=base_config.dtype,
            model_quantization_encoding=base_config.model_quantization_encoding,
            quantization_config=base_config.quantization_config,
            max_seq_len=base_config.max_seq_len,
            kv_params=olmo2_kv_params,  # Use Olmo2-specific KV params
            attention_multiplier=olmo2_attention_multiplier,  # Use Olmo2-specific attention multiplier
            embedding_multiplier=base_config.embedding_multiplier,
            residual_multiplier=base_config.residual_multiplier,
            devices=base_config.devices,
            clip_qkv=base_config.clip_qkv,
            use_subgraphs=base_config.use_subgraphs,
            lora_config=base_config.lora_config,
            logits_scaling=base_config.logits_scaling,
            data_parallel_degree=base_config.data_parallel_degree,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm",
        attention_bias: bool = False,
    ) -> None:
        """Define parameters that can't be determined just from the pipeline config.

        Delegates to the parent Llama3Config.finalize() method.

        Args:
            huggingface_config: The HuggingFace model configuration object.
            state_dict: The model's state dictionary containing weights.
            return_logits: Whether to return the last token, all tokens or a
                variable number of logits.
            return_hidden_states: Whether to return hidden states.
            norm_method: The normalization method to use.
            attention_bias: Whether to include bias in attention projections.
        """
        super().finalize(
            huggingface_config=huggingface_config,
            state_dict=state_dict,
            return_logits=return_logits,
            return_hidden_states=return_hidden_states,
            norm_method=norm_method,
            attention_bias=attention_bias,
        )
