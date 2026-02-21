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
"""Config for Idefics3 models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.legacy.kv_cache import KVCacheParams
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.architectures.llama3_legacy.model_config import Llama3Config
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from max.pipelines.lib.config_enums import supported_encoding_dtype
from max.pipelines.lib.interfaces.arch_config import ArchConfigWithKVCache
from transformers.models.auto.configuration_auto import AutoConfig
from typing_extensions import Self, override


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

    @classmethod
    def initialize_from_config(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        text_config_hidden_size: int,
    ) -> Idefics3VisionConfig:
        """Initialize Idefics3VisionConfig from HuggingFace config."""

        quantization_encoding = pipeline_config.model.quantization_encoding
        if quantization_encoding is None:
            raise ValueError("quantization_encoding must be set")
        dtype = supported_encoding_dtype(quantization_encoding)

        vision_config = getattr(huggingface_config, "vision_config", None)
        if vision_config is None:
            raise ValueError("vision_config not found in huggingface_config")
        scale_factor = getattr(huggingface_config, "scale_factor", 2)

        num_attention_heads = vision_config.num_attention_heads
        hidden_size = vision_config.hidden_size
        head_dim = hidden_size // num_attention_heads
        return cls(
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


@dataclass(kw_only=True)
class Idefics3Config(ArchConfigWithKVCache):
    """Configuration for Idefics3 models."""

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

    def get_kv_params(self) -> KVCacheParams:
        """Returns the KV cache parameters from the embedded text config."""
        return self.text_config.get_kv_params()

    def get_max_seq_len(self) -> int:
        """Returns the maximum sequence length from the embedded text config."""
        return self.text_config.get_max_seq_len()

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Get KV cache parameters for the language model."""
        # Delegate to Llama3Config for language model parameters.
        text_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        return Llama3Config.construct_kv_params(
            huggingface_config=text_config,
            pipeline_config=pipeline_config,
            devices=devices,
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

    @override
    @classmethod
    def initialize(cls, pipeline_config: PipelineConfig) -> Self:
        """Initializes an Idefics3Config instance from pipeline configuration.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.

        Returns:
            An Idefics3Config instance with fields initialized from config.
        """
        huggingface_config = pipeline_config.model.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for '{pipeline_config.model.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )

        # Create Llama3Config from the text config
        hf_text_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        text_config = Llama3Config.initialize_from_config(
            pipeline_config, hf_text_config
        )

        vision_config = Idefics3VisionConfig.initialize_from_config(
            pipeline_config, huggingface_config, text_config.hidden_size
        )

        return cls(
            devices=[
                DeviceRef(spec.device_type, spec.id)
                for spec in pipeline_config.model.device_specs
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

    def finalize(
        self,
        huggingface_config: AutoConfig,
        llm_state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
        norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm",
    ) -> None:
        """Finalize the Idefics3Config instance with state_dict dependent fields.

        Args:
            huggingface_config: HuggingFace model configuration.
            llm_state_dict: Language model weights dictionary.
            dtype: Data type for model parameters.
            return_logits: Return logits configuration.
            norm_method: Normalization method.
        """
        # Finalize text config
        hf_text_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        self.text_config.finalize(
            huggingface_config=hf_text_config,
            state_dict=llm_state_dict,
            norm_method=norm_method,
            attention_bias=False,
            return_logits=return_logits,
        )
