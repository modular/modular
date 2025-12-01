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
from typing import Literal

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn import ReturnLogits
from max.nn.kv_cache import KVCacheParams
from max.pipelines.architectures.llama3.model_config import (
    Llama3Config as Qwen2Config,
)
from max.pipelines.architectures.qwen3.model_config import Qwen3Config
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from transformers.models.auto.configuration_auto import AutoConfig


def _select_llm_config_class(
    hf_llm_cfg: AutoConfig,
) -> type[Qwen2Config | Qwen3Config]:
    """Choose the correct config class based on parameters in the HuggingFace
    config. Qwen2 is a wrapper around Llama3 and doesn't have its own config, so
    we alias Llama3Config as Qwen2Config for clarity."""
    mt = getattr(hf_llm_cfg, "model_type", None)
    archs = getattr(hf_llm_cfg, "architectures", None) or []
    if mt == "qwen3" or "Qwen3ForCausalLM" in archs:
        return Qwen3Config
    return Qwen2Config


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

    qkv_bias: bool
    """Whether to use bias in the QKV projection. Default: False."""

    o_proj_bias: bool
    """Whether to use bias in the out projection."""

    num_hidden_layers: int
    """Number of hidden layers in the vision encoder."""

    @staticmethod
    def generate(
        vision_config: AutoConfig,
        dtype: DType,
        state_dict: dict[str, WeightData],
    ) -> VisionConfig:
        """Generate VisionConfig from HuggingFace vision config.

        Args:
            vision_config: HuggingFace vision configuration object.
            state_dict: The model's state dictionary.

        Returns:
            Configured VisionConfig instance.
        """
        num_attention_heads = vision_config.num_attention_heads
        hidden_size = vision_config.hidden_size
        head_dim = hidden_size // num_attention_heads

        # InternVL o_proj_bias is not in the config, check checkpoint.
        # Check for the presence of the o_proj.bias key dynamically across all layers
        o_proj_bias = any(
            key.endswith(".attn.o_proj.bias") for key in state_dict
        )

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
            qkv_bias=getattr(vision_config, "qkv_bias", False),
            o_proj_bias=o_proj_bias,
            num_hidden_layers=getattr(vision_config, "num_hidden_layers", 32),
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
    llm_config: Qwen2Config | Qwen3Config
    """Language model configuration (Qwen2 or Qwen3)."""


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
        # Delegate to the selected decoder family for language model parameters.
        llm_hf_cfg = getattr(
            huggingface_config, "llm_config", huggingface_config
        )
        ConfigCls = _select_llm_config_class(llm_hf_cfg)
        return ConfigCls.get_kv_params(
            huggingface_config=llm_hf_cfg,
            n_devices=n_devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        # Delegate to the selected decoder family for language model parameters.
        llm_hf_cfg = getattr(
            huggingface_config, "llm_config", huggingface_config
        )
        ConfigCls = _select_llm_config_class(llm_hf_cfg)
        return ConfigCls.get_num_layers(llm_hf_cfg)

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Calculate maximum sequence length for InternVL."""
        # Delegate to the selected decoder family for language model parameters.
        llm_hf_cfg = getattr(
            huggingface_config, "llm_config", huggingface_config
        )
        ConfigCls = _select_llm_config_class(llm_hf_cfg)
        return ConfigCls.calculate_max_seq_len(
            pipeline_config=pipeline_config,
            huggingface_config=llm_hf_cfg,
        )

    @staticmethod
    def generate(
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        llm_state_dict: dict[str, WeightData],
        vision_state_dict: dict[str, WeightData],
        dtype: DType,
        n_devices: int,
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
            vision_state_dict: Vision model weights dictionary.
            dtype: Data type for model parameters.
            n_devices: Number of devices.
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
        vision_config = VisionConfig.generate(
            hf_vision_config, dtype, vision_state_dict
        )

        # Select decoder family (Qwen2/Qwen3) from HF llm_config
        hf_llm_config = getattr(
            huggingface_config, "llm_config", huggingface_config
        )
        ConfigCls = _select_llm_config_class(hf_llm_config)

        if ConfigCls is Qwen3Config:
            llm_config = Qwen3Config.generate(
                pipeline_config=pipeline_config,
                huggingface_config=hf_llm_config,
                state_dict=llm_state_dict,
                dtype=dtype,
                n_devices=n_devices,
                cache_dtype=cache_dtype,
                kv_cache_config=kv_cache_config,
                return_logits=return_logits,
                norm_method=norm_method,
                attention_bias=False,  # Qwen3 removes QKV biases
            )
        elif ConfigCls is Qwen2Config:
            # Qwen2 semantics (delegates to Llama3-style config under the hood)
            llm_config = Qwen2Config.generate(
                pipeline_config=pipeline_config,
                huggingface_config=hf_llm_config,
                state_dict=llm_state_dict,
                dtype=dtype,
                n_devices=n_devices,
                cache_dtype=cache_dtype,
                kv_cache_config=kv_cache_config,
                return_logits=return_logits,
                norm_method=norm_method,
                attention_bias=True,  # Qwen2 uses attention bias
            )  # type: ignore

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
