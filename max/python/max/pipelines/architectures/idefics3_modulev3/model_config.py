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
"""Config for Idefics3 models (ModuleV3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData, WeightsFormat, weights_format
from max.nn.kv_cache import KVCacheParams
from max.nn.rotary_embedding import Llama3RopeScalingParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.config_enums import supported_encoding_dtype
from max.pipelines.lib.interfaces.arch_config import ArchConfigWithKVCache
from transformers import AutoConfig
from typing_extensions import Self, override

# Reuse the vision config from the V2 implementation (no V3-specific changes).
from ..idefics3.model_config import Idefics3VisionConfig

# Use the V3 Llama3Config for the text component.
from ..llama3_modulev3.model_config import Llama3Config


@dataclass(kw_only=True)
class Idefics3Config(ArchConfigWithKVCache):
    """Configuration for Idefics3 models (ModuleV3)."""

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

    # Text model configuration - using V3 Llama3Config directly
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
        text_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        return Llama3Config.calculate_max_seq_len(
            pipeline_config=pipeline_config,
            huggingface_config=text_config,
        )

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
    ) -> Self:
        """Initializes an Idefics3Config instance from pipeline configuration."""
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for '{model_config.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )

        hf_text_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )

        # Build a V3 Llama3Config for the text component.
        text_config = _create_llama3_text_config(
            pipeline_config, hf_text_config
        )

        vision_config = Idefics3VisionConfig.initialize_from_config(
            pipeline_config, huggingface_config, text_config.hidden_size
        )

        return cls(
            devices=[
                DeviceRef(spec.device_type, spec.id)
                for spec in model_config.device_specs
            ],
            scale_factor=getattr(huggingface_config, "scale_factor", 2),
            image_token_id=getattr(
                huggingface_config, "image_token_id", 128257
            ),
            vision_config=vision_config,
            text_config=text_config,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        llm_state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm",
    ) -> None:
        """Finalize the Idefics3Config with state_dict-dependent fields."""
        hf_text_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        self.text_config.finalize(
            huggingface_config=hf_text_config,
            state_dict=llm_state_dict,
            norm_method=norm_method,
            attention_bias=False,
            return_logits=return_logits,
            return_hidden_states=return_hidden_states,
        )


def _create_llama3_text_config(
    pipeline_config: PipelineConfig,
    hf_text_config: AutoConfig,
) -> Llama3Config:
    """Create a V3 Llama3Config from a text sub-config.

    This replicates the initialization logic of Llama3Config.initialize()
    but accepts a specific HuggingFace text sub-config rather than reading
    from pipeline_config.model.huggingface_config.
    """
    kv_cache_config = pipeline_config.model.kv_cache
    quantization_encoding = pipeline_config.model.quantization_encoding
    if quantization_encoding is None:
        raise ValueError("quantization_encoding must not be None")
    dtype = supported_encoding_dtype(quantization_encoding)
    cache_dtype = pipeline_config.model.kv_cache.cache_dtype

    _weights_format = weights_format(pipeline_config.model.weight_path)
    interleaved_rope_weights = (
        _weights_format == WeightsFormat.gguf
        and pipeline_config.model.rope_type == "normal"
    )

    device_refs = [
        DeviceRef(spec.device_type, spec.id)
        for spec in pipeline_config.model.device_specs
    ]

    embedding_multiplier = getattr(hf_text_config, "embedding_multiplier", 1.0)
    residual_multiplier = getattr(hf_text_config, "residual_multiplier", 1.0)
    rope_scaling_params: Llama3RopeScalingParams | None = None
    rope_scaling = getattr(hf_text_config, "rope_scaling", None)

    if rope_scaling is not None:
        rope_type = rope_scaling.get("type")
        rope_type_alt = rope_scaling.get("rope_type")
        if rope_type == "llama3" or rope_type_alt == "llama3":
            rope_scaling_params = Llama3RopeScalingParams(
                factor=rope_scaling["factor"],
                low_freq_factor=rope_scaling["low_freq_factor"],
                high_freq_factor=rope_scaling["high_freq_factor"],
                orig_max_position=rope_scaling[
                    "original_max_position_embeddings"
                ],
            )

    attention_multiplier = Llama3Config.calculate_attention_multiplier(
        hf_text_config
    )

    max_seq_len = Llama3Config.calculate_max_seq_len(
        pipeline_config, huggingface_config=hf_text_config
    )

    return Llama3Config(
        hidden_size=hf_text_config.hidden_size,
        num_attention_heads=hf_text_config.num_attention_heads,
        num_key_value_heads=hf_text_config.num_key_value_heads,
        num_hidden_layers=hf_text_config.num_hidden_layers,
        rope_theta=hf_text_config.rope_theta,
        rope_scaling_params=rope_scaling_params,
        longrope_scaling_params=None,
        intermediate_size=hf_text_config.intermediate_size,
        interleaved_rope_weights=interleaved_rope_weights,
        vocab_size=hf_text_config.vocab_size,
        dtype=dtype,
        max_seq_len=max_seq_len,
        kv_params=Llama3Config.construct_kv_params(
            huggingface_config=hf_text_config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        ),
        attention_multiplier=attention_multiplier,
        embedding_multiplier=embedding_multiplier,
        residual_multiplier=residual_multiplier,
        devices=device_refs,
        clip_qkv=getattr(hf_text_config, "clip_qkv", None),
        logits_scaling=getattr(hf_text_config, "logits_scaling", 1.0),
    )
