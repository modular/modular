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
"""Config for Mistral models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.legacy.kv_cache import KVCacheParamInterface
from max.pipelines.architectures.mistral.model_config import MistralConfig
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.modeling.config_enums import SupportedEncoding
from transformers import AutoConfig
from typing_extensions import Self, override


@dataclass(kw_only=True)
class Mistral3Config(MistralConfig):
    """Configuration for Mistral3 models."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"bfloat16"}

    @override
    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        """Bounds against the text config's ``max_position_embeddings``."""
        huggingface_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        return super().calculate_max_seq_len(huggingface_config, model_config)

    @override
    @classmethod
    def construct_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
        *,
        allow_kv_head_replication: bool = False,
    ) -> KVCacheParamInterface:
        """Sizes the KV cache from the text config."""
        return super().construct_kv_params(
            getattr(huggingface_config, "text_config", huggingface_config),
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
            allow_kv_head_replication=allow_kv_head_replication,
        )

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Initializes a MistralConfig instance from pipeline configuration.

        This method creates a config instance with all fields that can be determined
        from the pipeline configuration.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            model_config: The MAX Engine model configuration; defaults to
                ``pipeline_config.model``.
            max_seq_len: The sequence length the cache was sized for.

        Returns:
            An initialized MistralConfig instance.
        """
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for '{model_config.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )
        return cls.initialize_from_config(
            pipeline_config,
            huggingface_config.text_config,
            max_seq_len=max_seq_len,
        )
