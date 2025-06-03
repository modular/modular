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

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParams
from max.pipelines.lib import (
    KVCacheConfig,
    MAXModelConfig,
    MAXModelConfigBase,
    PipelineConfig,
)
from transformers import AutoConfig


@dataclass
class InternVLConfigBase(MAXModelConfigBase):
    """Base configuration for InternVL models.

    Contains parameters specific to the InternVL architecture.
    """

    # Base config options.
    dtype: DType
    """DType of the model weights and input."""

    devices: list[DeviceRef]
    """Devices to run the model with."""

    max_seq_len: int
    """Maximum length of sequence."""

    # Vision encoder options.
    vision_hidden_size: int
    """Hidden size of the vision encoder."""

    image_size: int
    """Input image size."""

    patch_size: int
    """Vision transformer patch size."""

    # Language model options (from llm_config).
    hidden_size: int
    """Hidden size of the language model."""

    intermediate_size: int
    """Intermediate size in feed-forward layers."""

    num_attention_heads: int
    """Number of attention heads."""

    num_key_value_heads: int
    """Number of key-value heads."""

    num_hidden_layers: int
    """Number of transformer layers."""

    vocab_size: int
    """Vocabulary size."""

    # Multimodal options.
    downsample_ratio: float
    """Downsample ratio for vision features."""

    num_image_token: int
    """Number of image tokens per patch."""

    @staticmethod
    def help() -> dict[str, str]:
        """Returns a dictionary describing the configuration parameters."""
        # TODO: Populate this with helpful descriptions based on Args above.
        return {}


@dataclass
class InternVLConfig(MAXModelConfig, InternVLConfigBase):
    @staticmethod
    def get_kv_params(
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        # InternVL uses llm_config for language model parameters.
        text_config = huggingface_config.llm_config
        return KVCacheParams(
            page_size=kv_cache_config.kv_cache_page_size,
            dtype=cache_dtype,
            n_kv_heads=text_config.num_key_value_heads,
            head_dim=text_config.hidden_size // text_config.num_attention_heads,
            cache_strategy=kv_cache_config.cache_strategy,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
            enable_kvcache_swapping_to_host=kv_cache_config.enable_kvcache_swapping_to_host,
            host_kvcache_swap_space_gb=kv_cache_config.host_kvcache_swap_space_gb,
            n_devices=n_devices,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        # InternVL uses llm_config for language model parameters.
        return huggingface_config.llm_config.num_hidden_layers

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Calculate maximum sequence length for InternVL."""
        max_seq_len = pipeline_config.max_length
        if max_seq_len:
            return max_seq_len

        # Get `max_position_embeddings` from the `llm_config`.
        return huggingface_config.llm_config.max_position_embeddings

    @staticmethod
    def generate(
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        dtype: DType,
        n_devices: int,
    ) -> InternVLConfig:
        """Generate InternVLConfig from pipeline and HuggingFace configs.

        Args:
            pipeline_config: Pipeline configuration.
            huggingface_config: HuggingFace model configuration.
            state_dict: Model weights dictionary.
            dtype: Data type for model parameters.
            n_devices: Number of devices.

        Returns:
            Configured InternVLConfig instance.
        """
        # Get vision config
        vision_config = huggingface_config.vision_config

        # Get language model config
        llm_config = huggingface_config.llm_config

        # Device references
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model_config.device_specs
        ]

        return InternVLConfig(
            # Base config parameters
            dtype=dtype,
            devices=device_refs,
            max_seq_len=InternVLConfig.calculate_max_seq_len(
                pipeline_config, huggingface_config
            ),
            # Vision parameters
            vision_hidden_size=vision_config.hidden_size,
            image_size=vision_config.image_size,
            patch_size=vision_config.patch_size,
            # Language model parameters
            hidden_size=llm_config.hidden_size,
            intermediate_size=llm_config.intermediate_size,
            num_attention_heads=llm_config.num_attention_heads,
            num_key_value_heads=llm_config.num_key_value_heads,
            num_hidden_layers=llm_config.num_hidden_layers,
            vocab_size=llm_config.vocab_size,
            # Multimodal parameters
            downsample_ratio=getattr(
                huggingface_config, "downsample_ratio", 0.5
            ),
            num_image_token=getattr(huggingface_config, "num_image_token", 256),
        )
