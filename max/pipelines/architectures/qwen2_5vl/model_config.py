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
"""Config for Qwen2.5VL models."""

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
    """Base configuration for Qwen2.5VL models with required fields."""

    dtype: DType
    """DType of the Qwen2.5VL vision model weights."""

    devices: list[DeviceRef]
    """Devices that the Qwen2.5VL model is parallelized over."""

    patch_size: int
    """Vision transformer patch size."""

    temporal_patch_size: int
    """Vision transformer temporal patch size."""

    in_channels: int
    """Vision transformer number of input channels."""

    hidden_size: int
    """Hidden size of the vision encoder."""

    num_attention_heads: int
    """Number of attention heads in the vision encoder."""

    depth: int
    """Number of vision transformer layers."""

    intermediate_size: int
    """Intermediate size in the vision encoder's feed-forward layers."""

    out_hidden_size: int
    """Output hidden size of the vision encoder. Also the hidden size of the language model."""

    fullatt_block_indexes: list[int]
    """Indexes of the full attention blocks in the vision encoder."""

    rms_norm_eps: float
    """Epsilon for layer normalization."""

    window_size: int
    """Window size for the vision encoder."""

    spatial_merge_size: int
    """Spatial merge size for the vision encoder."""

    @staticmethod
    def generate(
        vision_config: AutoConfig,
        dtype: DType,
        pipeline_config: PipelineConfig,
    ) -> VisionConfig:
        """Generate VisionConfig from HuggingFace vision config.

        Args:
            vision_config: HuggingFace vision configuration object.

        Returns:
            Configured VisionConfig instance.
        """
        return VisionConfig(
            dtype=dtype,
            devices=[
                DeviceRef(spec.device_type, spec.id)
                for spec in pipeline_config.model_config.device_specs
            ],
            patch_size=vision_config.patch_size,
            temporal_patch_size=vision_config.temporal_patch_size,
            in_channels=vision_config.in_channels,
            hidden_size=vision_config.hidden_size,
            num_attention_heads=vision_config.num_heads,
            depth=vision_config.depth,
            intermediate_size=vision_config.intermediate_size,
            out_hidden_size=vision_config.out_hidden_size,
            fullatt_block_indexes=vision_config.fullatt_block_indexes,
            # TODO: fix this later
            rms_norm_eps=1e-06,
            window_size=vision_config.window_size,
            spatial_merge_size=vision_config.spatial_merge_size,
        )


@dataclass
class Qwen2_5VLConfigBase:
    """Base configuration for Qwen2.5VL models with required fields."""

    devices: list[DeviceRef]
    """Devices that the Qwen2.5VL model is parallelized over."""

    # Multimodal parameters
    image_token_id: int
    """Token ID used for image placeholders in the input sequence."""

    video_token_id: int
    """Token ID used for video placeholders in the input sequence."""

    vision_start_token_id: int
    """Token ID that marks the start of vision content."""

    spatial_merge_size: int
    """Size parameter for spatial merging of vision features."""

    tokens_per_second: int
    """Number of tokens per second."""

    mrope_section: list[int]
    """List of indices for the mrope section."""

    # Vision encoder configuration.
    vision_config: VisionConfig
    """Vision encoder configuration."""

    # Composed language model configuration.
    llm_config: Llama3Config
    """Language model configuration using Llama3 architecture."""


@dataclass
class Qwen2_5VLConfig(MAXModelConfig, Qwen2_5VLConfigBase):
    """Implementation of MAXModelConfig for Qwen2.5VL models."""

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
        """Calculate maximum sequence length for Qwen2.5VL."""
        # Delegate to Llama3Config for language model parameters.
        llm_config = getattr(
            huggingface_config, "text_config", huggingface_config
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
    ) -> Qwen2_5VLConfig:
        """Generate Qwen2_5VLConfig from pipeline and HuggingFace configs.

        Args:
            pipeline_config: Pipeline configuration.
            huggingface_config: HuggingFace model configuration.
            llm_state_dict: Model weights dictionary.
            vision_state_dict: Vision model weights dictionary.
            dtype: Data type for model parameters.
            n_devices: Number of devices.
            logits_postprocessor: Optional logits postprocessor.
            cache_dtype: KV cache data type.
            kv_cache_config: KV cache configuration.
            return_logits: Return logits configuration.
            norm_method: Normalization method.

        Returns:
            Configured Qwen2_5VLConfig instance.
        """
        # Create VisionConfig from the vision config
        hf_vision_config = getattr(huggingface_config, "vision_config", None)
        if hf_vision_config is None:
            raise ValueError("vision_config not found in huggingface_config")
        vision_config = VisionConfig.generate(
            hf_vision_config,
            dtype,
            pipeline_config,
        )

        # Create Llama3Config for the language model (with Qwen2 attention_bias=True)
        llm_config = Llama3Config.generate(
            pipeline_config=pipeline_config,
            huggingface_config=huggingface_config,
            state_dict=llm_state_dict,
            dtype=dtype,
            n_devices=n_devices,
            logits_postprocessor=logits_postprocessor,
            cache_dtype=cache_dtype,
            kv_cache_config=kv_cache_config,
            return_logits=return_logits,
            norm_method=norm_method,
            attention_bias=True,  # Qwen2.5VL uses Qwen2 which has attention_bias=True
        )

        return Qwen2_5VLConfig(
            devices=[
                DeviceRef(spec.device_type, spec.id)
                for spec in pipeline_config.model_config.device_specs
            ],
            # Multimodal parameters
            image_token_id=huggingface_config.image_token_id,
            video_token_id=huggingface_config.video_token_id,
            vision_start_token_id=huggingface_config.vision_start_token_id,
            spatial_merge_size=hf_vision_config.spatial_merge_size,
            tokens_per_second=hf_vision_config.tokens_per_second,
            mrope_section=huggingface_config.rope_scaling["mrope_section"],
            # Vision configuration
            vision_config=vision_config,
            # Composed language model configuration
            llm_config=llm_config,
        )
