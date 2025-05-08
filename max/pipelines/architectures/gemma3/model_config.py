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
from typing import Callable, Literal

from max.dtype import DType
from max.graph import DeviceRef, TensorValue
from max.graph.weights import WeightData, WeightsFormat, weights_format
from max.nn import LinearScalingParams, ReturnLogits
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
class Gemma3ConfigBase(MAXModelConfigBase):
    """Base configuration for Gemma 3 models.

    Contains parameters specific to the Gemma 3 architecture, typically
    extracted from a HuggingFace configuration object's text config.
    """

    # Gemma 3 specific parameters (taken from Transformer's `configuration_gemma3.py`)
    vocab_size: int
    """Vocabulary size of the Gemma3Text model."""

    hidden_size: int
    """Dimension of the hidden representations."""

    intermediate_size: int
    """Dimension of the MLP representations."""

    num_hidden_layers: int
    """Number of hidden layers in the Transformer decoder."""

    num_attention_heads: int
    """Number of attention heads for each attention layer in the Transformer
    decoder."""

    num_key_value_heads: int
    """Number of key_value heads that should be used to implement Grouped Query
    Attention."""

    head_dim: int
    """The attention head dimension."""

    hidden_activation: str
    """The non-linear activation function (function or string) in the decoder.
    Will default to `"gelu_tanh"` if not specified. `"gelu_tanh"`
    uses an approximation of the `"gelu"` activation function."""

    max_position_embeddings: int
    """The maximum sequence length that this model might ever be used with."""

    rms_norm_eps: float
    """The epsilon used by the rms normalization layers."""

    tie_word_embeddings: bool
    """Whether to tie weight embeddings. When true, the output linear layer
    uses the same
    weight as the embedding layer."""

    rope_theta: float
    """The base period of the RoPE embeddings."""

    attention_bias: bool
    """Whether to use a bias in the query, key, value and output projection
    layers during self-attention."""

    query_pre_attn_scalar: float | None
    """Scaling factor used on the attention scores."""

    sliding_window: int
    """In the Gemma3 language model, every other layer uses sliding window
    attention. This is the size of the sliding window."""

    final_logit_softcapping: float | None
    """Scaling factor when applying tanh softcapping on the logits."""

    attn_logit_softcapping: int | None
    """Scaling factor when applying tanh softcapping on the attention scores."""

    rope_scaling: LinearScalingParams | None
    """Scaling configuration for the RoPE embeddings used in global attention."""

    rope_local_base_freq: float
    """The base period of the RoPE embeddings for local attention."""

    sliding_window_pattern: int
    """Pattern for the sliding window attention."""

    # Max-specific config parameters.
    dtype: DType
    """DType of the model weights and input."""

    devices: list[DeviceRef]
    """Devices to run the model with."""

    interleaved_rope_weights: bool
    """True if the rope weights are in interleaved complex format."""

    return_logits: ReturnLogits
    """Whether to return the last token, all logits, or a variable number of logits."""

    kv_params: KVCacheParams
    """KV cache parameters."""


@dataclass
class Gemma3Config(MAXModelConfig, Gemma3ConfigBase):
    """Represents the complete MAX Engine configuration for Gemma 3 models.

    Combines the base Gemma 3 parameters with MAX-specific settings and
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
        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=huggingface_config.head_dim,
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
        return huggingface_config.num_hidden_layers

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
        return huggingface_config.max_position_embeddings

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
        norm_method: Literal["rms_norm"] = "rms_norm",
        attention_bias: bool = False,  # Gemma3 attention bias is False in HF.
    ) -> Gemma3Config:
        """Generates a Gemma3Config instance from various configuration sources.

        This factory method takes pipeline settings, HuggingFace configuration,
        model state dictionary, and other parameters to construct a fully initialized
        :obj:`Gemma3Config` object for use within the MAX Engine pipeline.

        Args:
            pipeline_config: The MAX Engine pipeline configuration (:obj:`max.pipelines.config.PipelineConfig`).
            huggingface_config: The HuggingFace model configuration object (:obj:`transformers.AutoConfig`).
            state_dict: The model's state dictionary containing weights (:obj:`max.graph.weights.WeightData`).
            dtype: The primary data type for model parameters (:obj:`max.dtype.DType`).
            n_devices: The number of devices the model will run on.
            logits_postprocessor: An optional callable to post-process model logits (:obj:`max.graph.TensorValue`).
            cache_dtype: The data type for the KV cache (:obj:`max.dtype.DType`).
            kv_cache_config: Configuration settings for the KV cache (:obj:`max.pipelines.max_config.KVCacheConfig`).
            return_logits: Whether to return the last token, all tokens or a variable number of logits.
            norm_method: The normalization method to use (currently only "rms_norm").
            attention_bias: Whether to include bias in attention projections. Defaults
              to `False` based on Gemma 3 HuggingFace implementation.

        Returns:
            An initialized :obj:`Gemma3Config` instance.
        """
        _weights_format = weights_format(
            pipeline_config.model_config.weight_path
        )
        interleaved_rope_weights = (
            _weights_format == WeightsFormat.gguf
            and pipeline_config.model_config.rope_type == RopeType.normal
        )
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model_config.device_specs
        ]

        # When tie_word_embeddings=True, the embedding weights are shared with
        # the output weights.
        tie_word_embeddings = (
            getattr(huggingface_config, "tie_word_embeddings", False)
            or "language_model.lm_head.weight" not in state_dict
        )

        rope_scaling_params = None
        rope_scaling = huggingface_config.rope_scaling

        if rope_scaling is not None:
            # Since "rope_type" huggingface config is not standardized, we need
            # to check for both "type" and "rope_type" keys.
            rope_type = rope_scaling.get("type")
            rope_type_alt = rope_scaling.get("rope_type")
            if rope_type is None and rope_type_alt is None:
                raise ValueError(
                    "Neither 'type' nor 'rope_type' found in rope_scaling huggingface config"
                )
            if rope_type == "linear" or rope_type_alt == "linear":
                rope_scaling_params = LinearScalingParams(
                    factor=rope_scaling["factor"]
                )

        hidden_activation = _HIDDEN_ACTIVATION_MAP.get(
            huggingface_config.hidden_activation,
            huggingface_config.hidden_activation,
        )

        return Gemma3Config(
            vocab_size=huggingface_config.vocab_size,
            hidden_size=huggingface_config.hidden_size,
            intermediate_size=huggingface_config.intermediate_size,
            num_hidden_layers=huggingface_config.num_hidden_layers,
            num_attention_heads=huggingface_config.num_attention_heads,
            num_key_value_heads=huggingface_config.num_key_value_heads,
            head_dim=huggingface_config.head_dim,
            hidden_activation=hidden_activation,
            max_position_embeddings=huggingface_config.max_position_embeddings,
            rms_norm_eps=huggingface_config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=huggingface_config.rope_theta,
            attention_bias=huggingface_config.attention_bias,
            query_pre_attn_scalar=huggingface_config.query_pre_attn_scalar,
            sliding_window=huggingface_config.sliding_window,
            final_logit_softcapping=huggingface_config.final_logit_softcapping,
            attn_logit_softcapping=huggingface_config.attn_logit_softcapping,
            rope_scaling=rope_scaling_params,
            rope_local_base_freq=huggingface_config.rope_local_base_freq,
            sliding_window_pattern=huggingface_config.sliding_window_pattern,
            dtype=dtype,
            devices=device_refs,
            interleaved_rope_weights=interleaved_rope_weights,
            return_logits=return_logits,
            kv_params=Gemma3Config.get_kv_params(
                huggingface_config=huggingface_config,
                n_devices=n_devices,
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
        )


_HIDDEN_ACTIVATION_MAP = {
    "gelu_pytorch_tanh": "gelu_tanh",
    "swish": "silu",
}
