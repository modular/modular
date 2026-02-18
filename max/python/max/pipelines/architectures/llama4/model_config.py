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

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.legacy.kv_cache import KVCacheParams
from max.nn.legacy.rotary_embedding import Llama3RopeScalingParams
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.lib import KVCacheConfig, PipelineConfig, RopeType
from max.pipelines.lib.interfaces.arch_config import ArchConfigWithKVCache
from transformers import AutoConfig
from typing_extensions import Self, override


@dataclass(kw_only=True)
class Llama4Config(ArchConfigWithKVCache):
    """Configuration for Llama 4 models.

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

    vocab_size: int
    """Size of the vocabulary."""

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

    tie_word_embeddings: bool = False
    """Whether to tie input and output word embeddings."""

    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    """Whether to return the last token, all logits, or a variable number of logits."""

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
        """Constructs the KV cache parameters from configuration objects.

        Args:
            huggingface_config: The HuggingFace model configuration object (:obj:`transformers.AutoConfig`).
            devices: The list of devices the model will run on.
            kv_cache_config: The MAX Engine KV cache configuration settings (:obj:`max.pipelines.max_config.KVCacheConfig`).
            cache_dtype: The desired data type for the KV cache (:obj:`max.dtype.DType`).

        Returns:
            The configured :obj:`max.pipelines.kv_cache.KVCacheParams` object.
        """
        text_config = huggingface_config.text_config
        return kv_cache_config.to_params(
            dtype=cache_dtype,
            n_kv_heads=text_config.num_key_value_heads,
            head_dim=text_config.head_dim,
            num_layers=Llama4Config.get_num_layers(huggingface_config),
            devices=devices,
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
        max_seq_len = pipeline_config.model.max_length
        if max_seq_len:
            return max_seq_len

        # Access max_position_embeddings from the text_config.
        return huggingface_config.text_config.max_position_embeddings

    @override
    @classmethod
    def initialize(cls, pipeline_config: PipelineConfig) -> Self:
        """Initializes a Llama4Config instance from pipeline configuration.

        This method creates a config instance with all fields that can be determined
        from the pipeline configuration, without needing the state_dict.
        Fields that depend on the state_dict (like tie_word_embeddings)
        should be set via the `finalize()` method.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.

        Returns:
            An initialized Llama4Config instance.
        """
        huggingface_config = pipeline_config.model.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for '{pipeline_config.model.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )
        kv_cache_config = pipeline_config.model.kv_cache
        quantization_encoding = pipeline_config.model.quantization_encoding
        if quantization_encoding is None:
            raise ValueError("quantization_encoding must not be None")
        dtype = quantization_encoding.dtype
        cache_dtype = pipeline_config.model.kv_cache.cache_dtype

        interleaved_rope_weights = (
            pipeline_config.model.rope_type == RopeType.normal
        )
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model.device_specs
        ]

        text_config = huggingface_config.text_config

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

        kv_params = cls.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

        return cls(
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
            vocab_size=text_config.vocab_size,
            max_seq_len=cls.calculate_max_seq_len(
                pipeline_config, huggingface_config
            ),
            num_hidden_layers=text_config.num_hidden_layers,
            kv_params=kv_params,
            dtype=dtype,
            devices=device_refs,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
    ) -> None:
        """Define parameters that can't be determined just from the pipeline config.

        Args:
            huggingface_config: The HuggingFace model configuration object.
            state_dict: The model's state dictionary containing weights.
            return_logits: Whether to return the last token, all tokens or a variable number of logits.
        """
        # When tie_word_embeddings=True, the embedding weights are shared with
        # the output weights.
        tie_word_embeddings = (
            getattr(huggingface_config, "tie_word_embeddings", False)
            or "lm_head.weight" not in state_dict
        )

        self.tie_word_embeddings = tie_word_embeddings
        self.return_logits = return_logits
