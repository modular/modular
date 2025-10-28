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

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn import Llama3RopeScalingParams, ReturnLogits
from max.nn.kv_cache import KVCacheParams
from max.pipelines.lib import (
    KVCacheConfig,
    MAXModelConfig,
    MAXModelConfigBase,
    PipelineConfig,
    RopeType,
)
from transformers import AutoConfig


@dataclass
class Llama4ConfigBase(MAXModelConfigBase):
    """Base configuration for Llama 4 models.

    Contains parameters specific to the Llama 4 architecture, typically
    extracted from a HuggingFace configuration object's text config.
    """

    # Llama 4 specific parameters (extracted from hf_config.text_config).
    hidden_size: int
    """Dimensionality of the embedding and attention layers."""

    intermediate_size: int
    """Dimensionality of the intermediate layer in feed-forward blocks."""

    intermediate_size_mlp: int
    """Dimensionality of the intermediate layer in dense MLP blocks."""

    num_attention_heads: int
    """Number of attention heads."""

    num_key_value_heads: int
    """Number of key/value heads (for Grouped Query Attention)."""

    head_dim: int
    """Dimensionality of each attention head."""

    rope_theta: float
    """Base period for RoPE embeddings."""

    rope_scaling_params: Llama3RopeScalingParams | None
    """Configuration for RoPE scaling."""

    interleaved_rope_weights: bool
    """True if the rope weights are in interleaved complex format."""

    num_experts_per_tok: int
    """Number of experts to route to per token in MoE layers."""

    num_local_experts: int
    """Total number of experts available in MoE layers."""

    moe_layers: list[int]
    """List of layer indices that are MoE layers."""

    interleave_moe_layer_step: int
    """Step size for interleaving MoE layers."""

    use_qk_norm: bool
    """Whether to apply L2 normalization to query and key states."""

    no_rope_layer_interval: int
    """Interval for skipping RoPE application."""

    attention_chunk_size: int
    """Chunk size for attention computation."""

    attn_temperature_tuning: bool
    """Whether to enable infernece-time temperature tuning attention in NoRoPE
    layers. This is useful for very long contexts."""

    floor_scale: int
    """Scaling factor used in attention temperature tuning calculation."""

    attn_scale: float
    """Scaling factor for attention scores."""

    rms_norm_eps: float
    """Epsilon value for RMS normalization layers."""

    tie_word_embeddings: bool
    """Whether to tie input and output word embeddings."""

    vocab_size: int
    """Size of the vocabulary."""

    return_logits: ReturnLogits
    """Whether to return the last token, all logits, or a variable number of logits."""

    max_seq_len: int
    """Maximum length of sequence."""

    num_hidden_layers: int
    """Number of decoder layers in the model."""

    kv_params: KVCacheParams
    """KV cache parameters."""

    dtype: DType
    """DType of the model weights and input."""

    devices: list[DeviceRef]
    """Devices to run the model with."""

    @staticmethod
    def help() -> dict[str, str]:
        """Returns a dictionary describing the configuration parameters."""
        # TODO: Populate this with helpful descriptions based on Args above.
        return {}


@dataclass
class Llama4Config(MAXModelConfig, Llama4ConfigBase):
    """Represents the complete MAX Engine configuration for Llama 4 models.

    Combines the base Llama 4 parameters with MAX-specific settings and
    provides methods to derive necessary pipeline components like KV cache parameters.
    """

    @staticmethod
    def get_kv_params(
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Constructs the KV cache parameters from configuration objects.

        Args:
            huggingface_config: The HuggingFace model configuration object (:obj:`transformers.AutoConfig`).
            n_devices: The number of devices the model will run on.
            kv_cache_config: The MAX Engine KV cache configuration settings (:obj:`max.pipelines.max_config.KVCacheConfig`).
            cache_dtype: The desired data type for the KV cache (:obj:`max.dtype.DType`).

        Returns:
            The configured :obj:`max.pipelines.kv_cache.KVCacheParams` object.
        """
        text_config = huggingface_config.text_config
        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=text_config.num_key_value_heads,
            head_dim=text_config.head_dim,
            page_size=kv_cache_config.kv_cache_page_size,
            cache_strategy=kv_cache_config.cache_strategy,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
            enable_kvcache_swapping_to_host=kv_cache_config.enable_kvcache_swapping_to_host,
            host_kvcache_swap_space_gb=kv_cache_config.host_kvcache_swap_space_gb,
            n_devices=n_devices,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        """Retrieves the number of hidden layers from the HuggingFace configuration.

        Args:
            huggingface_config: The HuggingFace model configuration object (:obj:`transformers.AutoConfig`).

        Returns:
            The number of hidden layers specified in the configuration's text config.
        """
        return huggingface_config.text_config.num_hidden_layers

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Calculates the maximum sequence length for the model.

        Uses the `max_length` from the :obj:`max.pipelines.config.PipelineConfig` if provided,
        otherwise falls back to the `max_position_embeddings` from the HuggingFace
        configuration's text config.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            huggingface_config: The HuggingFace model configuration object (:obj:`transformers.AutoConfig`).

        Returns:
            The calculated maximum sequence length.
        """
        max_seq_len = pipeline_config.max_length
        if max_seq_len:
            return max_seq_len

        # Access max_position_embeddings from the text_config.
        return huggingface_config.text_config.max_position_embeddings

    @staticmethod
    def generate(
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        dtype: DType,
        n_devices: int,
        cache_dtype: DType,
        kv_cache_config: KVCacheConfig,
        return_logits: ReturnLogits,
        norm_method: Literal["rms_norm"] = "rms_norm",
        attention_bias: bool = False,  # Llama4 attention bias is False in HF.
    ) -> Llama4Config:
        """Generates a Llama4Config instance from various configuration sources.

        This factory method takes pipeline settings, HuggingFace configuration,
        model state dictionary, and other parameters to construct a fully initialized
        :obj:`Llama4Config` object for use within the MAX Engine pipeline.

        Args:
            pipeline_config: The MAX Engine pipeline configuration (:obj:`max.pipelines.config.PipelineConfig`).
            huggingface_config: The HuggingFace model configuration object (:obj:`transformers.AutoConfig`).
            state_dict: The model's state dictionary containing weights (:obj:`max.graph.weights.WeightData`).
            dtype: The primary data type for model parameters (:obj:`max.dtype.DType`).
            n_devices: The number of devices the model will run on.
            cache_dtype: The data type for the KV cache (:obj:`max.dtype.DType`).
            kv_cache_config: Configuration settings for the KV cache (:obj:`max.pipelines.max_config.KVCacheConfig`).
            return_logits: Whether to return the last token, all tokens or a variable number of logits.
            norm_method: The normalization method to use (currently only "rms_norm").
            attention_bias: Whether to include bias in attention projections. Defaults
              to `False` based on Llama 4 HuggingFace implementation.

        Returns:
            An initialized :obj:`Llama4Config` instance.
        """
        interleaved_rope_weights = (
            pipeline_config.model_config.rope_type == RopeType.normal
        )
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model_config.device_specs
        ]

        text_config = huggingface_config.text_config

        # When tie_word_embeddings=True, the embedding weights are shared with
        # the output weights.
        tie_word_embeddings = (
            getattr(huggingface_config, "tie_word_embeddings", False)
            or "lm_head.weight" not in state_dict
        )

        rope_scaling_params = None
        rope_scaling = text_config.rope_scaling

        if rope_scaling is not None:
            # Since "rope_type" huggingface config is not standardized, we need
            # to check for both "type" and "rope_type" keys.
            rope_type = rope_scaling.get("type")
            rope_type_alt = rope_scaling.get("rope_type")
            if rope_type is None and rope_type_alt is None:
                raise ValueError(
                    "Neither 'type' nor 'rope_type' found in rope_scaling huggingface config"
                )
            if rope_type == "llama3" or rope_type_alt == "llama3":
                rope_scaling_params = Llama3RopeScalingParams(
                    factor=rope_scaling["factor"],
                    low_freq_factor=rope_scaling["low_freq_factor"],
                    high_freq_factor=rope_scaling["high_freq_factor"],
                    orig_max_position=rope_scaling[
                        "original_max_position_embeddings"
                    ],
                )

        return Llama4Config(
            hidden_size=text_config.hidden_size,
            intermediate_size=text_config.intermediate_size,
            intermediate_size_mlp=text_config.intermediate_size_mlp,
            num_attention_heads=text_config.num_attention_heads,
            num_key_value_heads=text_config.num_key_value_heads,
            head_dim=text_config.head_dim,
            rope_theta=text_config.rope_theta,
            rope_scaling_params=rope_scaling_params,
            interleaved_rope_weights=interleaved_rope_weights,
            num_experts_per_tok=text_config.num_experts_per_tok,
            num_local_experts=text_config.num_local_experts,
            moe_layers=list(
                range(
                    text_config.interleave_moe_layer_step - 1,
                    text_config.num_hidden_layers,
                    text_config.interleave_moe_layer_step,
                )
            ),
            interleave_moe_layer_step=text_config.interleave_moe_layer_step,
            use_qk_norm=text_config.use_qk_norm,
            no_rope_layer_interval=getattr(
                text_config, "no_rope_layer_interval", 4
            ),
            attention_chunk_size=text_config.attention_chunk_size,
            attn_temperature_tuning=getattr(
                text_config, "attn_temperature_tuning", True
            ),
            floor_scale=getattr(text_config, "floor_scale", 8192),
            attn_scale=getattr(text_config, "attn_scale", 0.1),
            rms_norm_eps=text_config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            vocab_size=text_config.vocab_size,
            return_logits=return_logits,
            max_seq_len=Llama4Config.calculate_max_seq_len(
                pipeline_config, huggingface_config
            ),
            num_hidden_layers=text_config.num_hidden_layers,
            kv_params=Llama4Config.get_kv_params(
                huggingface_config=huggingface_config,
                n_devices=n_devices,
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            dtype=dtype,
            devices=device_refs,
        )
