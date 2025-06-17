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
"""Config for InternVL models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from max.dtype import DType
from max.graph import DeviceRef, TensorValue
from max.graph.weights import WeightData
from max.nn import ReturnLogits
from max.nn.kv_cache import KVCacheParams
from max.pipelines.architectures.llama3.model_config import Llama3Config
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from transformers.models.auto.configuration_auto import AutoConfig


@dataclass
class VisionConfig:
    """Base configuration for InternVL models with required fields."""

    dtype: DType
    """DType of the InternVL vision model weights."""

    hidden_size: int
    """Hidden size of the vision encoder."""

    intermediate_size: int
    """Intermediate size in the vision encoder's feed-forward layers."""

    norm_type: Literal["rms_norm"] | Literal["layer_norm"]
    """Type of normalization used in the vision encoder."""

    image_size: int
    """Input image size."""

    patch_size: int
    """Vision transformer patch size."""

    num_attention_heads: int
    """Number of attention heads in the vision encoder."""

    head_dim: int
    """Dimension of each attention head."""

    layer_norm_eps: float
    """Epsilon for layer normalization."""

    qk_normalization: bool
    """Whether to use QK normalization in attention."""

    num_hidden_layers: int
    """Number of hidden layers in the vision encoder."""

    use_mean_pooling: bool
    """Whether to use mean pooling instead of final layernorm."""

    @staticmethod
    def generate(vision_config: AutoConfig, dtype: DType) -> VisionConfig:
        """Generate VisionConfig from HuggingFace vision config.

        Args:
            vision_config: HuggingFace vision configuration object.

        Returns:
            Configured VisionConfig instance.
        """
        num_attention_heads = vision_config.num_attention_heads
        hidden_size = vision_config.hidden_size
        head_dim = hidden_size // num_attention_heads

        return VisionConfig(
            dtype=dtype,
            hidden_size=hidden_size,
            intermediate_size=vision_config.intermediate_size,
            norm_type=getattr(vision_config, "norm_type", "rms_norm"),
            image_size=vision_config.image_size,
            patch_size=vision_config.patch_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            layer_norm_eps=getattr(vision_config, "layer_norm_eps", 1e-6),
            qk_normalization=getattr(vision_config, "qk_normalization", True),
            num_hidden_layers=getattr(vision_config, "num_hidden_layers", 32),
            use_mean_pooling=getattr(
                vision_config, "use_mean_pooling", True
            ),  # Default to True for InternVL
        )


@dataclass
class InternVLConfigBase:
    """Base configuration for InternVL models with required fields."""

    devices: list[DeviceRef]
    """Devices that the InternVL model is parallelized over."""

    # Multimodal options.
    downsample_ratio: float
    """Downsample ratio for vision features."""

    num_image_token: int
    """Number of image tokens per patch."""

    # Vision encoder configuration.
    vision_config: VisionConfig
    """Vision encoder configuration."""

    # Composed language model configuration.
    llm_config: Llama3Config
    """Language model configuration using Llama3 architecture."""


@dataclass
class InternVLConfig(MAXModelConfig, InternVLConfigBase):
    """Implementation of MAXModelConfig for InternVL models."""

    @staticmethod
    def help() -> dict[str, str]:
        """Returns a dictionary describing the configuration parameters."""
        # TODO: Populate this with helpful descriptions based on Args above.
        return {}

    @staticmethod
    def get_kv_params(
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        # Delegate to Llama3Config for language model parameters.
        llm_config = getattr(
            huggingface_config, "llm_config", huggingface_config
        )
        return Llama3Config.get_kv_params(
            huggingface_config=llm_config,
            n_devices=n_devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        # Delegate to Llama3Config for language model parameters.
        llm_config = getattr(
            huggingface_config, "llm_config", huggingface_config
        )
        return Llama3Config.get_num_layers(llm_config)

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Calculate maximum sequence length for InternVL."""
        # Delegate to Llama3Config for language model parameters.
        llm_config = getattr(
            huggingface_config, "llm_config", huggingface_config
        )
        return Llama3Config.calculate_max_seq_len(
            pipeline_config=pipeline_config,
            huggingface_config=llm_config,
        )

    @staticmethod
    def generate(
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        llm_state_dict: dict[str, WeightData],
        dtype: DType,
        n_devices: int,
        logits_postprocessor: Callable[[TensorValue], TensorValue] | None,
        cache_dtype: DType,
        kv_cache_config: KVCacheConfig,
        return_logits: ReturnLogits,
        norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm",
    ) -> InternVLConfig:
        """Generate InternVLConfig from pipeline and HuggingFace configs.

        Args:
            pipeline_config: Pipeline configuration.
            huggingface_config: HuggingFace model configuration.
            llm_state_dict: Model weights dictionary.
            dtype: Data type for model parameters.
            n_devices: Number of devices.
            logits_postprocessor: Optional logits postprocessor.
            cache_dtype: KV cache data type.
            kv_cache_config: KV cache configuration.
            return_logits: Return logits configuration.
            norm_method: Normalization method.

        Returns:
            Configured InternVLConfig instance.
        """
        # Create VisionConfig from the vision config
        hf_vision_config = getattr(huggingface_config, "vision_config", None)
        if hf_vision_config is None:
            raise ValueError("vision_config not found in huggingface_config")
        vision_config = VisionConfig.generate(hf_vision_config, dtype)

        # Create Llama3Config for the language model (with Qwen2 attention_bias=True)
        hf_llm_config = getattr(
            huggingface_config, "llm_config", huggingface_config
        )
        llm_config = Llama3Config.generate(
            pipeline_config=pipeline_config,
            huggingface_config=huggingface_config.llm_config,
            state_dict=llm_state_dict,
            dtype=dtype,
            n_devices=n_devices,
            logits_postprocessor=logits_postprocessor,
            cache_dtype=cache_dtype,
            kv_cache_config=kv_cache_config,
            return_logits=return_logits,
            norm_method=norm_method,
            attention_bias=True,  # InternVL uses Qwen2 which has attention_bias=True
        )

        return InternVLConfig(
            devices=[
                DeviceRef(spec.device_type, spec.id)
                for spec in pipeline_config.model_config.device_specs
            ],
            # Multimodal parameters
            downsample_ratio=getattr(
                huggingface_config, "downsample_ratio", 0.5
            ),
            num_image_token=getattr(huggingface_config, "num_image_token", 256),
            # Vision configuration
            vision_config=vision_config,
            # Composed language model configuration
            llm_config=llm_config,
        )
