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
"""Qwen3Embedding configuration."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.quantization import QuantizationEncoding
from max.graph.weights import WeightData
from max.pipelines.lib import MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfig,
    ModelOutputConfig,
)
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from transformers import AutoConfig
from typing_extensions import Self


@dataclass(kw_only=True)
class Qwen3EmbeddingConfig(ArchConfig):
    """Qwen3 embedding model configuration."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "float32",
        "bfloat16",
    }

    pipeline_config: PipelineConfig
    max_seq_len: int
    quantization_encoding: SupportedEncoding | None = None

    embedding_quantization: QuantizationEncoding | None = None
    """The checkpoint's token-embedding encoding, when quantized (GGUF),
    which :meth:`finalize` reads."""

    @property
    def huggingface_config(self) -> AutoConfig:
        """The checkpoint's config."""
        config = self.pipeline_config.model.huggingface_config
        assert config is not None
        return config

    @property
    def dtype(self) -> DType:
        """The model's dtype, from the resolved quantization encoding."""
        return supported_encoding_dtype(
            _select_quantization_encoding(
                self.pipeline_config.model, self.DEFAULT_ENCODING
            )
        )

    @property
    def devices(self) -> list[DeviceRef]:
        """The devices the model runs on."""
        return [
            DeviceRef(spec.device_type, spec.id)
            for spec in self.pipeline_config.model.device_specs
        ]

    @property
    def pool_embeddings(self) -> bool:
        """Whether the model returns pooled, normalized embeddings."""
        return self.pipeline_config.model.pool_embeddings

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: Mapping[str, WeightData],
        outputs: ModelOutputConfig,
    ) -> None:
        """Records the token embedding's encoding from the checkpoint."""
        del huggingface_config, outputs
        embed_weight = state_dict.get("embed_tokens.weight")
        self.embedding_quantization = (
            embed_weight.quantization_encoding if embed_weight else None
        )

    def get_max_seq_len(self) -> int:
        """The maximum sequence length the config allows."""
        return self.max_seq_len

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        """The configured ``max_length``, bounded by the checkpoint's."""
        # The configured max_length, bounded by max_position_embeddings.
        model_max = getattr(
            huggingface_config, "max_position_embeddings", 32768
        )
        configured_max = model_config.max_length or 8192

        if configured_max > model_max:
            raise ValueError(
                f"Configured max_length ({configured_max}) exceeds model's "
                f"max_position_embeddings ({model_max})"
            )

        return configured_max

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Builds the config from the pipeline configuration."""
        return cls(pipeline_config=pipeline_config, max_seq_len=max_seq_len)
