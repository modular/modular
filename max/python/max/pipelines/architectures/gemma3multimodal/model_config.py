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

from dataclasses import dataclass

from max.dtype.dtype import DType
from max.nn import ReturnLogits
from max.nn.kv_cache import KVCacheParams
from max.pipelines.architectures.gemma3.model_config import Gemma3Config
from max.pipelines.lib import (
    KVCacheConfig,
    PipelineConfig,
)
from transformers.models.auto.configuration_auto import AutoConfig


@dataclass
class VisionConfig:
    hidden_size: int
    image_size: int
    intermediate_size: int
    model_type: str
    num_attention_heads: int
    num_hidden_layers: int
    patch_size: int
    vision_use_head: bool
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
    dtype: DType = DType.bfloat16

    @staticmethod
    def generate(hf_config: AutoConfig) -> "VisionConfig":
        return VisionConfig(
            hidden_size=hf_config.hidden_size,
            image_size=hf_config.image_size,
            intermediate_size=hf_config.intermediate_size,
            model_type=hf_config.model_type,
            num_attention_heads=hf_config.num_attention_heads,
            num_hidden_layers=hf_config.num_hidden_layers,
            patch_size=hf_config.patch_size,
            vision_use_head=hf_config.vision_use_head,
            num_channels=getattr(hf_config, "num_channels", 3),
            layer_norm_eps=getattr(hf_config, "layer_norm_eps", 1e-6),
        )


@dataclass
class Gemma3MultimodalConfigBase:
    """Base configuration for Gemma3 multimodal models."""

    vision_config: VisionConfig
    text_config: Gemma3Config
    boi_token_index: int
    eoi_token_index: int
    eos_token_id: list[int]
    image_token_index: int
    initializer_range: float
    mm_tokens_per_image: int
    model_type: str
    torch_dtype: DType


class Gemma3MultimodalConfig(Gemma3MultimodalConfigBase):
    # TODO: fix the return type
    @staticmethod
    def generate(
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        state_dict: dict,
        dtype: DType,
        n_devices: int,
        cache_dtype: DType,
        kv_cache_config: KVCacheConfig,
        return_logits: ReturnLogits,
        attention_bias: bool = False,
    ) -> "Gemma3MultimodalConfig":
        # Check if this is a multimodal config or language-only config
        hf_vision_config = getattr(huggingface_config, "vision_config", None)
        hf_text_config = getattr(huggingface_config, "text_config", None)

        # If no text_config, use the config itself as the text config (language-only model)
        if hf_text_config is None:
            hf_text_config = huggingface_config

        # Generate text config
        text_config = Gemma3Config.generate(
            pipeline_config=pipeline_config,
            huggingface_config=hf_text_config,
            state_dict=state_dict,
            dtype=dtype,
            n_devices=n_devices,
            cache_dtype=cache_dtype,
            kv_cache_config=kv_cache_config,
            return_logits=return_logits,
            attention_bias=attention_bias,
        )

        if hf_vision_config is not None:
            vision_config = VisionConfig.generate(hf_vision_config)
        else:
            raise ValueError(
                "No vision_config found in HuggingFace config. "
                "This model may not be a multimodal Gemma3 model."
            )

        return Gemma3MultimodalConfig(
            vision_config=vision_config,
            text_config=text_config,
            boi_token_index=getattr(huggingface_config, "boi_token_index", 0),
            eoi_token_index=getattr(huggingface_config, "eoi_token_index", 0),
            eos_token_id=getattr(huggingface_config, "eos_token_id", []),
            image_token_index=getattr(
                huggingface_config, "image_token_index", 0
            ),
            initializer_range=getattr(
                huggingface_config, "initializer_range", 0.02
            ),
            mm_tokens_per_image=getattr(
                huggingface_config, "mm_tokens_per_image", 256
            ),
            model_type=getattr(huggingface_config, "model_type", "gemma3"),
            torch_dtype=getattr(huggingface_config, "torch_dtype", dtype),
        )

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
        text_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        return text_config.max_position_embeddings

    @staticmethod
    def get_kv_params(
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        text_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
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
        text_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        return text_config.num_hidden_layers
