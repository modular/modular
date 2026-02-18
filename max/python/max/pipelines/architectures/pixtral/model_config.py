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
from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.legacy.kv_cache import KVCacheParams
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from max.pipelines.lib.interfaces.arch_config import ArchConfigWithKVCache
from transformers import AutoConfig
from typing_extensions import Self, override


@dataclass(kw_only=True)
class PixtralConfig(ArchConfigWithKVCache):
    """Configuration for Pixtral models."""

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
    kv_params: KVCacheParams
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

    def get_kv_params(self) -> KVCacheParams:
        return self.kv_params

    def get_max_seq_len(self) -> int:
        return self.max_seq_len

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return kv_cache_config.to_params(
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.text_config.num_key_value_heads,
            head_dim=huggingface_config.text_config.head_dim,
            num_layers=PixtralConfig.get_num_layers(huggingface_config),
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        return huggingface_config.text_config.num_hidden_layers

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Calculates the maximum sequence length for the model."""
        max_seq_len = pipeline_config.model.max_length
        if max_seq_len:
            return max_seq_len
        return huggingface_config.text_config.max_position_embeddings

    @override
    @classmethod
    def initialize(cls, pipeline_config: PipelineConfig) -> Self:
        """Initializes a PixtralConfig instance from pipeline configuration.

        This method creates a config instance with all fields that can be determined
        from the pipeline configuration.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.

        Returns:
            An initialized PixtralConfig instance.
        """
        huggingface_config = pipeline_config.model.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for '{pipeline_config.model.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )
        kv_cache_config = pipeline_config.model.kv_cache
        quantization_encoding = pipeline_config.model.quantization_encoding
        if quantization_encoding is None:
            raise ValueError("quantization_encoding must not be None")
        dtype = quantization_encoding.dtype
        cache_dtype = pipeline_config.model.kv_cache.cache_dtype

        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model.device_specs
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
            rope_theta=text_config.rope_theta,
            max_seq_len=cls.calculate_max_seq_len(
                pipeline_config, huggingface_config
            ),
            num_hidden_layers=text_config.num_hidden_layers,
            head_dim=text_config.head_dim,
            num_key_value_heads=text_config.num_key_value_heads,
            feed_forward_length=text_config.intermediate_size,
            vocab_size=text_config.vocab_size,
            kv_params=kv_params,
            attention_multiplier=math.sqrt(1 / kv_params.head_dim),
            patch_size=vision_config.patch_size,
            image_size=vision_config.image_size,
            num_channels=vision_config.num_channels,
            vision_hidden_size=vision_config.hidden_size,
            vision_num_attention_heads=vision_config.num_attention_heads,
            vision_rope_theta=vision_config.rope_theta,
            vision_num_hidden_layers=vision_config.num_hidden_layers,
            vision_intermediate_size=vision_config.intermediate_size,
            vision_head_dim=vision_config.head_dim,
        )
