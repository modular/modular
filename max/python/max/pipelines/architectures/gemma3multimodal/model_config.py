from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from transformers import AutoConfig

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData, WeightsFormat, weights_format
from max.nn import LinearScalingParams, ReturnLogits
from max.nn.float8_config import Float8Config, parse_float8_config
from max.nn.kv_cache import KVCacheParams
from max.pipelines.lib import (
    MAXModelConfig,
    MAXModelConfigBase,
    KVCacheConfig,
    PipelineConfig,
    RopeType,
)

@dataclass
class SiglipVisionConfig:
    """
    The vision-specific config for Gemma3
    fields and defaults taken from below link - unsure if they are valid
    https://huggingface.co/google/gemma-3-4b-it/blob/main/config.json
    """

    hidden_act: str | None
    """The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
    `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported."""
    
    hidden_size: int
    """Dimensionality of the encoder layers and the pooler layer"""

    image_size: int
    """The size (resolution) of each image"""

    intermediate_size: int
    """maybe not required?"""

    layer_norm_eps: float | None
    """The epsilon used by the layer normalization layers."""

    num_attention_heads: int
    """Number of attention heads for each attention layer in the Transformer encoder"""

    num_hidden_layers: int
    """Number of hidden layers in the Transformer encoder"""

    num_channels: int | None
    """Number of channels in the input images."""

    patch_size: int
    """The size (resolution) of each patch"""

    model_type: str = "siglip_vision_model"
    """model type for AutoConfig"""

    vision_use_head: bool = False
    """maybe not required?
    Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder"""

    @staticmethod
    def generate(
        vision_config: AutoConfig
    ) -> SiglipVisionConfig:
        return SiglipVisionConfig(
            hidden_size=vision_config.hidden_size,
            image_size=vision_config.image_size,
            intermediate_size=vision_config.intermediate_size,
            num_attention_heads=vision_config.num_attention_heads,
            num_hidden_layers=getattr(vision_config, "num_hidden_layers", 32),
            patch_size=vision_config.patch_size,
            num_channels=vision_config.num_channels,
            hidden_act=vision_config.hidden_act,
            layer_norm_eps=vision_config.layer_norm_eps,
        )
    
@dataclass
class Gemma3TextConfig:    
    attention_bias: bool
    """Whether to use a bias in the query, key, value and output projection
    layers during self-attention."""

    attn_logit_softcapping: int | None
    """Scaling factor when applying tanh softcapping on the attention scores."""

    sliding_window: int
    """In the Gemma3 language model, every other layer uses sliding window
    attention. This is the size of the sliding window."""

    _sliding_window_pattern: int # TODO required?
    """transformers/models/gemma3/configuration_gemma.py no idea what it does or what the value represents"""

    final_logit_softcapping: float | None
    """Scaling factor when applying tanh softcapping on the logits."""

    head_dim: int
    """The attention head dimension."""

    hidden_activation: str
    """The non-linear activation function (function or string) in the decoder.
    Will default to `"gelu_tanh"` if not specified. `"gelu_tanh"`
    uses an approximation of the `"gelu"` activation function."""
    
    hidden_size: int
    """Dimension of the hidden representations."""

    initializer_range: float # TODO figure out.  in Text and overall config??
    """The standard deviation of the truncated_normal_initializer for initializing all weight matrices"""
    
    intermediate_size: int
    """Dimension of the MLP representations."""

    layer_types: list[str] | None
    """Attention pattern for each layer, optional"""
    
    max_position_embeddings: int
    """The maximum sequence length that this model might ever be used with."""
    
    num_hidden_layers: int
    """Number of hidden layers in the Transformer decoder."""

    num_attention_heads: int
    """Number of attention heads for each attention layer in the Transformer
    decoder."""

    num_key_value_heads: int
    """Number of key_value heads that should be used to implement Grouped Query
    Attention."""

    query_pre_attn_scalar: float | None
    """Scaling factor used on the attention scores."""
    
    rms_norm_eps: float
    """The epsilon used by the rms normalization layers."""
    
    rope_scaling: LinearScalingParams | None
    """Scaling configuration for the RoPE embeddings used in global attention."""

    rope_local_base_freq: float
    """The base period of the RoPE embeddings for local attention."""
    
    rope_theta: float
    """The base period of the RoPE embeddings."""

    vocab_size: int
    """Vocabulary size of the Gemma3Text model."""

    attention_dropout: float = 0.0
    """The dropout ratio for the attention probabilities.  Optional, defaults to 0.0"""

    use_bidirectional_attention: bool = False
    """If True, the model will attend to all text tokens instead of using a causal mask. This does not change
    behavior for vision tokens."""

    use_cache: bool = True
    """Whether or not the model should return the last key/values attentions (not used by all models). Only
    relevant if `config.is_decoder=True`"""

    @staticmethod
    def generate(
        text_config: AutoConfig
    ) -> Gemma3TextConfig:
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
            if rope_type == "linear" or rope_type_alt == "linear":
                rope_scaling_params = LinearScalingParams(
                    factor=rope_scaling["factor"]
                )

        hidden_activation = _HIDDEN_ACTIVATION_MAP.get(
            text_config.hidden_activation,
            text_config.hidden_activation,
        )
        
        return Gemma3TextConfig(
            sliding_window=text_config.sliding_window,
            attention_bias=text_config.attention_bias,
            _sliding_window_pattern=6, # TODO no idea.  came from transformers code comments
            layer_types=text_config.layer_types,
            initializer_range=text_config.initializer_range,
            use_bidirectional_attention=text_config.use_bidirectional_attention,
            use_cache=text_config.use_cache,
            attn_logit_softcapping=text_config.attn_logit_softcapping,
            final_logit_softcapping=text_config.final_logit_softcapping,
            head_dim=text_config.head_dim,
            hidden_activation=hidden_activation,
            hidden_size=text_config.hidden_size,
            intermediate_size=text_config.intermediate_size,
            max_position_embeddings=text_config.max_position_embeddings,
            num_hidden_layers=text_config.num_hidden_layers,
            num_attention_heads=text_config.num_attention_heads,
            num_key_value_heads=text_config.num_key_value_heads,
            query_pre_attn_scalar=text_config.query_pre_attn_scalar,
            rms_norm_eps=text_config.rms_norm_eps,
            rope_scaling=rope_scaling_params,
            rope_local_base_freq=text_config.rope_local_base_freq,
            rope_theta=text_config.rope_theta,
            vocab_size=text_config.vocab_size,
        )
    
@dataclass
class Gemma3MultiModalConfigBase(MAXModelConfigBase):
    """Base configuration for Gemma 3 models.

    Contains parameters specific to the Gemma 3 architecture, typically
    extracted from a HuggingFace configuration object's text config.
    """

    boi_token_index: int
    """The begin-of-image token index to wrap the image prompt"""

    eoi_token_index: int
    """The end-of-image token index to wrap the image prompt"""

    devices: list[DeviceRef]
    """Devices to run the model with."""

    dtype: DType
    """DType of the model weights and input."""

    kv_params: KVCacheParams
    """KV cache parameters."""

    image_token_index: int
    """The image token index to encode the image prompt"""

    initializer_range: float # TODO figure out.  in Text and overall config??

    interleaved_rope_weights: bool
    """True if the rope weights are in interleaved complex format."""

    mm_tokens_per_image: int
    """The number of tokens per image embedding"""

    return_logits: ReturnLogits
    """Whether to return the last token, all logits, or a variable number of logits."""

    tie_word_embeddings: bool
    """Whether to tie weight embeddings. When true, the output linear layer
    uses the same
    weight as the embedding layer."""

    text_config: Gemma3TextConfig
    """The config object of the text backbone"""
    
    # https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/gemma3/configuration_gemma3.py
    vision_config: SiglipVisionConfig
    """Custom vision config or dict"""

    model_type: str = "Gemma3ForConditionalGeneration"
    """the name of the model type for auto config"""

    float8_config: Float8Config | None = None
    """Float8 quantization configuration."""

@dataclass
class Gemma3ForConditionalGenerationConfig(MAXModelConfig, Gemma3MultiModalConfigBase):
    @staticmethod
    def get_kv_params(
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.text_config.num_key_value_heads,
            head_dim=huggingface_config.text_config.head_dim,
            page_size=kv_cache_config.kv_cache_page_size,
            cache_strategy=kv_cache_config.cache_strategy,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
            enable_kvcache_swapping_to_host=kv_cache_config.enable_kvcache_swapping_to_host,
            host_kvcache_swap_space_gb=kv_cache_config.host_kvcache_swap_space_gb,
            n_devices=n_devices,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        return huggingface_config.text_config.num_hidden_layers # TODO text?  vision?  who can tell

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        max_seq_len = pipeline_config.max_length
        if max_seq_len:
            return max_seq_len
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
        attention_bias: bool = False,  # Gemma3 attention bias is False in HF.
    ) -> Gemma3ForConditionalGenerationConfig:
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

        # Parse the float8 config from compressed-tensors
        layer_name_prefix = "language_model."
        float8_config = parse_float8_config(
            huggingface_config,
            state_dict,
            dtype,
            state_dict_name_prefix=layer_name_prefix,
            ignored_modules_prefix=layer_name_prefix,
        )

        # override SiglipVisionConfig and Gemma3TextConfig from the huggingface AutoConfig
        hf_vision_config = getattr(huggingface_config, "vision_config", None)
        if hf_vision_config is None:
            raise ValueError("vision_config not found in huggingface_config")
        vision_config = SiglipVisionConfig.generate(hf_vision_config)

        hf_text_config = getattr(huggingface_config, "text_config", None)
        if hf_text_config is None:
            raise ValueError("text_config not found in huggingface_config")
        text_config = Gemma3TextConfig.generate(hf_text_config)

        gemma3_config = Gemma3ForConditionalGenerationConfig(
            tie_word_embeddings=tie_word_embeddings,
            dtype=dtype,
            devices=device_refs,
            interleaved_rope_weights=interleaved_rope_weights,
            return_logits=return_logits,
            kv_params=Gemma3ForConditionalGenerationConfig.get_kv_params(
                huggingface_config=huggingface_config,
                n_devices=n_devices,
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            float8_config=float8_config,
            
            vision_config=vision_config,
            text_config=text_config,
            mm_tokens_per_image=huggingface_config.mm_tokens_per_image,
            boi_token_index=huggingface_config.boi_token_index,
            eoi_token_index=huggingface_config.eoi_token_index,
            image_token_index=huggingface_config.image_token_index,
            initializer_range=0.0,
        )

        gemma3_config.mm_tokens_per_image = huggingface_config.mm_tokens_per_image
        gemma3_config.boi_token_index = huggingface_config.boi_token_index
        gemma3_config.eoi_token_index = huggingface_config.eoi_token_index
        gemma3_config.image_token_index = huggingface_config.image_token_index
        gemma3_config.initializer_range = huggingface_config.initializer_range

        return gemma3_config

_HIDDEN_ACTIVATION_MAP = {
    "gelu_pytorch_tanh": "gelu_tanh",
    "swish": "silu",
}