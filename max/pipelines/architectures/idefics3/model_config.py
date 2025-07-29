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
"""Config for Idefics3 models."""

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
class Idefics3VisionConfig:
    """Configuration for Idefics3 Vision Model (SigLIP-based)."""

    dtype: DType
    """DType of the Idefics3 vision model weights."""

    hidden_size: int
    """Hidden size of the vision encoder."""

    intermediate_size: int
    """Intermediate size in the vision encoder's feed-forward layers."""

    image_size: int
    """Input image size."""

    patch_size: int
    """Vision transformer patch size."""

    num_channels: int
    """Number of input channels (typically 3 for RGB)."""

    num_attention_heads: int
    """Number of attention heads in the vision encoder."""

    head_dim: int
    """Dimension of each attention head."""

    layer_norm_eps: float
    """Epsilon for layer normalization."""

    hidden_act: str
    """Activation function used in the vision encoder."""

    num_hidden_layers: int
    """Number of hidden layers in the vision encoder."""

    initializer_range: float
    """Standard deviation for weight initialization."""

    scale_factor: int
    """Scale factor for pixel shuffle operation in the connector."""

    text_config_hidden_size: int
    """Hidden size from the text config for modality projection."""

    @staticmethod
    def generate(
        vision_config: AutoConfig,
        dtype: DType,
        scale_factor: int,
        text_config_hidden_size: int,
    ) -> Idefics3VisionConfig:
        """Generate Idefics3VisionConfig from HuggingFace vision config.

        Args:
            vision_config: HuggingFace vision configuration object.
            dtype: Data type for the vision model.
            scale_factor: Scale factor for pixel shuffle operation in the connector.
            text_config_hidden_size: Hidden size from the text config for modality projection.

        Returns:
            Configured Idefics3VisionConfig instance.
        """
        num_attention_heads = vision_config.num_attention_heads
        hidden_size = vision_config.hidden_size
        head_dim = hidden_size // num_attention_heads

        return Idefics3VisionConfig(
            dtype=dtype,
            hidden_size=hidden_size,
            intermediate_size=vision_config.intermediate_size,
            image_size=vision_config.image_size,
            patch_size=vision_config.patch_size,
            num_channels=getattr(vision_config, "num_channels", 3),
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            layer_norm_eps=getattr(vision_config, "layer_norm_eps", 1e-6),
            hidden_act=getattr(
                vision_config, "hidden_act", "gelu_pytorch_tanh"
            ),
            num_hidden_layers=vision_config.num_hidden_layers,
            initializer_range=getattr(vision_config, "initializer_range", 0.02),
            scale_factor=scale_factor,
            text_config_hidden_size=text_config_hidden_size,
        )


@dataclass
class Idefics3ConfigBase:
    """Base configuration for Idefics3 models with required fields."""

    devices: list[DeviceRef]
    """Devices that the Idefics3 model is parallelized over."""

    # Multimodal options.
    scale_factor: int
    """Scale factor for pixel shuffle operation in the connector."""

    image_token_id: int
    """Token ID used to represent image tokens in the text sequence."""

    # Vision encoder configuration.
    vision_config: Idefics3VisionConfig
    """Vision encoder configuration (SigLIP-based)."""

    # Text model configuration - using Llama3Config directly
    text_config: Llama3Config
    """Text model configuration (Llama3-based)."""

    @property
    def image_seq_len(self) -> int:
        """Calculate the number of image tokens after connector processing."""
        patches_per_side = (
            self.vision_config.image_size // self.vision_config.patch_size
        )
        total_patches = patches_per_side * patches_per_side
        return total_patches // (self.scale_factor * self.scale_factor)


@dataclass
class Idefics3Config(MAXModelConfig, Idefics3ConfigBase):
    """Implementation of MAXModelConfig for Idefics3 models."""

    @staticmethod
    def help() -> dict[str, str]:
        """Returns a dictionary describing the configuration parameters."""
        return {
            "scale_factor": "Factor by which spatial resolution is reduced in pixel shuffle",
            "image_token_id": "Special token ID representing image patches in text sequence",
            "vision_config": "Configuration for the SigLIP-based vision encoder",
            "text_config": "Configuration for the Llama3-based text model",
        }

    @staticmethod
    def get_kv_params(
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Get KV cache parameters for the language model."""
        # Delegate to Llama3Config for language model parameters.
        text_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        return Llama3Config.get_kv_params(
            huggingface_config=text_config,
            n_devices=n_devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        """Get number of layers in the language model."""
        text_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        return text_config.num_hidden_layers

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Calculate maximum sequence length for Idefics3."""
        # Delegate to Llama3Config for language model parameters.
        text_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        return Llama3Config.calculate_max_seq_len(
            pipeline_config=pipeline_config,
            huggingface_config=text_config,
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
    ) -> Idefics3Config:
        """Generate Idefics3Config from pipeline and HuggingFace configs.

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
            Configured Idefics3Config instance.
        """
        # Create Llama3Config from the text config first to get the hidden size
        hf_text_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        text_config = Llama3Config.generate(
            pipeline_config=pipeline_config,
            huggingface_config=hf_text_config,
            state_dict=llm_state_dict,
            dtype=dtype,
            n_devices=n_devices,
            logits_postprocessor=logits_postprocessor,
            cache_dtype=cache_dtype,
            kv_cache_config=kv_cache_config,
            return_logits=return_logits,
            norm_method=norm_method,
        )

        # Create Idefics3VisionConfig from the vision config
        hf_vision_config = getattr(huggingface_config, "vision_config", None)
        if hf_vision_config is None:
            raise ValueError("vision_config not found in huggingface_config")
        scale_factor = getattr(huggingface_config, "scale_factor", 2)
        vision_config = Idefics3VisionConfig.generate(
            hf_vision_config, dtype, scale_factor, text_config.hidden_size
        )

        return Idefics3Config(
            devices=[
                DeviceRef(spec.device_type, spec.id)
                for spec in pipeline_config.model_config.device_specs
            ],
            # Multimodal parameters specific to Idefics3
            scale_factor=getattr(huggingface_config, "scale_factor", 2),
            image_token_id=getattr(
                huggingface_config, "image_token_id", 128257
            ),
            # Vision configuration (SigLIP-based)
            vision_config=vision_config,
            # Text model configuration (Llama3-based)
            text_config=text_config,
        )
