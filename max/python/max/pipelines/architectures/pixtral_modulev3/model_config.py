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
"""Config for Pixtral models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParamInterface
from max.nn.outputs import ReturnLogits
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.graph_specs import GraphRole
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithKVCache,
    ArchConfigWithStoredKVParams,
    ArchConfigWithVisionCache,
    ArchVLConfigWithTextSubconfig,
    ModelOutputConfig,
)
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta
from max.pipelines.lib.registration_runtime import DeploymentConfig
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from transformers import AutoConfig
from typing_extensions import Self, override


@dataclass(kw_only=True)
class PixtralConfig(
    ArchVLConfigWithTextSubconfig,
    ArchConfigWithStoredKVParams,
    ArchConfigWithKVCache,
    ArchConfigWithVisionCache,
    DeploymentConfig,
):
    """Configuration for Pixtral models."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"bfloat16"}

    dtype: DType
    devices: list[DeviceRef]

    # Llava fields
    image_token_index: int

    # Language model fields
    hidden_size: int
    num_attention_heads: int
    rms_norm_eps: float
    rope_theta: float
    max_seq_len: int
    num_hidden_layers: int
    head_dim: int
    num_key_value_heads: int
    feed_forward_length: int
    vocab_size: int
    kv_params: KVCacheParamInterface
    attention_multiplier: float

    # Vision encoder fields
    patch_size: int
    image_size: int
    num_channels: int
    vision_hidden_size: int
    vision_num_attention_heads: int
    vision_rope_theta: float
    vision_num_hidden_layers: int
    vision_intermediate_size: int
    vision_head_dim: int

    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    """Whether to return the last token, all logits, or a variable number of logits."""

    quantization_encoding: SupportedEncoding | None = None

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        """The language model's hidden layer count."""
        return huggingface_config.text_config.num_hidden_layers

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Initializes a PixtralConfig instance from pipeline configuration.

        This method creates a config instance with all fields that can be determined
        from the pipeline configuration.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            model_config: The MAX model configuration to read the HuggingFace
                config from. Defaults to ``pipeline_config.model``.
            max_seq_len: The maximum sequence length the config allows.

        Returns:
            An initialized PixtralConfig instance.
        """
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for '{model_config.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )
        if model_config.enable_echo:
            raise ValueError(
                "Pixtral model does not currently implement enable echo."
            )
        if len(model_config.device_specs) != 1:
            raise ValueError("Pixtral does not support distributed inference")
        kv_cache_config = model_config.kv_cache
        quantization_encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        dtype = supported_encoding_dtype(quantization_encoding)
        cache_dtype = cache_dtype_for_encoding(
            quantization_encoding, model_config.kv_cache.kv_cache_format
        )

        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs
        ]

        kv_params = cls.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

        text_config = huggingface_config.text_config
        vision_config = huggingface_config.vision_config

        return cls(
            dtype=dtype,
            devices=device_refs,
            image_token_index=huggingface_config.image_token_index,
            hidden_size=text_config.hidden_size,
            num_attention_heads=text_config.num_attention_heads,
            rms_norm_eps=text_config.rms_norm_eps,
            rope_theta=get_rope_theta(text_config),
            max_seq_len=max_seq_len,
            num_hidden_layers=text_config.num_hidden_layers,
            head_dim=text_config.head_dim,
            num_key_value_heads=text_config.num_key_value_heads,
            feed_forward_length=text_config.intermediate_size,
            vocab_size=text_config.vocab_size,
            kv_params=kv_params,
            attention_multiplier=math.sqrt(1 / text_config.head_dim),
            patch_size=vision_config.patch_size,
            image_size=vision_config.image_size,
            num_channels=vision_config.num_channels,
            vision_hidden_size=vision_config.hidden_size,
            vision_num_attention_heads=vision_config.num_attention_heads,
            vision_rope_theta=get_rope_theta(vision_config),
            vision_num_hidden_layers=vision_config.num_hidden_layers,
            vision_intermediate_size=vision_config.intermediate_size,
            vision_head_dim=vision_config.head_dim,
            quantization_encoding=quantization_encoding,
        )

    weight_prefixes: ClassVar[Mapping[str, str]] = {
        GraphRole.DECODER: "decoder.",
        GraphRole.VISION: "vision.",
    }

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: Mapping[str, object],
        outputs: ModelOutputConfig,
    ) -> Self:
        """Applies the requested logits mode before constructing either tower."""
        del huggingface_config, state_dict
        return replace(self, return_logits=outputs.return_logits)

    @classmethod
    def get_vision_cache_row_spec(
        cls, huggingface_config: AutoConfig
    ) -> tuple[int, DType]:
        """Returns the projected image embedding width and dtype."""
        return huggingface_config.text_config.hidden_size, DType.bfloat16

    @classmethod
    def estimate_vision_cache_entry_bytes(
        cls, huggingface_config: AutoConfig
    ) -> int:
        """Sizes one image's projected patch rows."""
        vision = huggingface_config.vision_config
        width, dtype = cls.get_vision_cache_row_spec(huggingface_config)
        return (
            (vision.image_size // vision.patch_size) ** 2
            * width
            * dtype.size_in_bytes
        )
