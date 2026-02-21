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
"""Configuration for Bert models."""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.config_enums import supported_encoding_dtype
from max.pipelines.lib.interfaces.arch_config import ArchConfig
from max.pipelines.lib.utils import upper_bounded_default
from transformers import AutoConfig
from typing_extensions import Self, override


@dataclass(kw_only=True)
class BertModelConfig(ArchConfig):
    """Configuration for Bert models."""

    dtype: DType
    device: DeviceRef
    pool_embeddings: bool
    huggingface_config: AutoConfig
    pipeline_config: PipelineConfig

    @override
    def get_max_seq_len(self) -> int:
        try:
            return upper_bounded_default(
                upper_bound=self.huggingface_config.max_position_embeddings,
                default=self.pipeline_config.model.max_length,
            )
        except ValueError as e:
            raise ValueError(
                "Unable to infer max_length for Bert, the provided "
                f"max_length ({self.pipeline_config.model.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({self.huggingface_config.max_position_embeddings})."
            ) from e

    @override
    @classmethod
    def initialize(cls, pipeline_config: PipelineConfig) -> Self:
        """Initializes a BertModelConfig instance from pipeline configuration.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.

        Returns:
            An initialized BertModelConfig instance.
        """
        quantization_encoding = pipeline_config.model.quantization_encoding
        if quantization_encoding is None:
            raise ValueError("quantization_encoding must not be None")
        if len(pipeline_config.model.device_specs) != 1:
            raise ValueError("BERT model is only supported on a single device")
        device_spec = pipeline_config.model.device_specs[0]
        huggingface_config = pipeline_config.model.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for '{pipeline_config.model.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )
        return cls(
            dtype=supported_encoding_dtype(quantization_encoding),
            device=DeviceRef(
                device_type=device_spec.device_type, id=device_spec.id
            ),
            pool_embeddings=pipeline_config.pool_embeddings,
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
        )
