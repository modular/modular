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
from max.graph.weights import WeightData, WeightsFormat, weights_format
from max.nn import ReturnLogits
from max.nn.float8_config import Float8Config, parse_float8_config
from max.nn.kv_cache import KVCacheParams
from max.pipelines.architectures.gemma3.model_config import Gemma3Config
from max.pipelines.lib import (
    KVCacheConfig,
    MAXModelConfig,
    MAXModelConfigBase,
    PipelineConfig,
    RopeType,
)
from transformers import AutoConfig

# handy https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/gemma3/configuration_gemma3.py


@dataclass
class Gemma3VisionConfig:
    """
    The vision-specific config for Gemma3
    fields and defaults taken from below link - unsure if they are valid
    https://huggingface.co/google/gemma-3-4b-it/blob/main/config.json
    """

    hidden_act: str | None
    """The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
    `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported."""

    hidden_size: int
    """Dimensionality of the encoder layers and the pooler layer.  default 1152"""

    image_size: int
    """The size (resolution) of each image.  default 896"""

    intermediate_size: int
    """maybe not required? default 4304"""

    layer_norm_eps: float | None
    """The epsilon used by the layer normalization layers."""

    num_attention_heads: int
    """Number of attention heads for each attention layer in the Transformer encoder.  default 16"""

    num_hidden_layers: int
    """Number of hidden layers in the Transformer encoder.  default 27"""

    num_channels: int | None
    """Number of channels in the input images."""

    patch_size: int
    """The size (resolution) of each patch.  default 14"""

    attention_dropout: float = 0.0
    """The dropout ratio for the attention probabilities"""

    model_type: str = "Gemma3"  # "siglip_vision_model" (HF)
    """model type for AutoConfig"""

    vision_use_head: bool = False
    """maybe not required?
    Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder"""

    @staticmethod
    def generate(vision_config: AutoConfig) -> Gemma3VisionConfig:
        return Gemma3VisionConfig(
            hidden_size=vision_config.hidden_size,
            image_size=vision_config.image_size,
            intermediate_size=vision_config.intermediate_size,
            num_attention_heads=vision_config.num_attention_heads,
            num_hidden_layers=vision_config.num_hidden_layers,
            patch_size=vision_config.patch_size,
            num_channels=vision_config.num_channels,
            hidden_act=vision_config.hidden_act,
            layer_norm_eps=vision_config.layer_norm_eps,
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

    initializer_range: float  # TODO figure out.  in Text and overall config??

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

    text_config: Gemma3Config
    """The config object of the text backbone"""

    # https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/gemma3/configuration_gemma3.py
    vision_config: Gemma3VisionConfig
    """Custom vision config or dict"""

    attention_bias: bool = False
    """Whether to use a bias in the query, key, value and output projection layers during self-attention."""

    attn_logit_softcapping: float = None
    """Scaling factor when applying tanh softcapping on the attention scores."""

    float8_config: Float8Config | None = None
    """Float8 quantization configuration."""

    head_dim: int = 256
    """The attention head dimension."""

    model_type: str = "Gemma3ForConditionalGeneration"
    """the name of the model type for auto config"""

    num_key_value_heads: int = 4
    """
    This is the number of key_value heads that should be used to implement Grouped Query Attention. If
    `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
    `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
    converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed"
    """


@dataclass
class Gemma3ForConditionalGenerationConfig(
    MAXModelConfig, Gemma3MultiModalConfigBase
):
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
        return (
            huggingface_config.text_config.num_hidden_layers
        )  # TODO text?  vision?  who can tell

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

        # override SiglipVisionConfig and Gemma3Config from the huggingface AutoConfig
        hf_vision_config = getattr(huggingface_config, "vision_config", None)
        if hf_vision_config is None:
            raise ValueError("vision_config not found in huggingface_config")
        vision_config = Gemma3VisionConfig.generate(hf_vision_config)

        hf_text_config = getattr(huggingface_config, "text_config", None)
        if hf_text_config is None:
            raise ValueError("text_config not found in huggingface_config")
        text_config = Gemma3Config.generate(
            pipeline_config=pipeline_config,
            huggingface_config=hf_text_config,
            state_dict=state_dict,
            dtype=dtype,
            n_devices=n_devices,
            cache_dtype=cache_dtype,
            kv_cache_config=kv_cache_config,
            return_logits=return_logits,
            norm_method=norm_method,
            attention_bias=attention_bias,
        )

        kv_params = Gemma3ForConditionalGenerationConfig.get_kv_params(
            huggingface_config=huggingface_config,
            n_devices=n_devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

        gemma3_config = Gemma3ForConditionalGenerationConfig(
            tie_word_embeddings=tie_word_embeddings,
            dtype=dtype,
            devices=device_refs,
            interleaved_rope_weights=interleaved_rope_weights,
            return_logits=return_logits,
            kv_params=kv_params,
            float8_config=float8_config,
            vision_config=vision_config,
            text_config=text_config,
            mm_tokens_per_image=huggingface_config.mm_tokens_per_image,
            boi_token_index=huggingface_config.boi_token_index,
            eoi_token_index=huggingface_config.eoi_token_index,
            image_token_index=huggingface_config.image_token_index,
            initializer_range=0.0,
        )

        return gemma3_config


_HIDDEN_ACTIVATION_MAP = {
    "gelu_pytorch_tanh": "gelu_tanh",
    "swish": "silu",
}
