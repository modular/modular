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

from max.pipelines.architectures.mistral.model_config import MistralConfig
from max.pipelines.lib import PipelineConfig
from typing_extensions import Self, override


@dataclass(kw_only=True)
class Mistral3Config(MistralConfig):
    """Configuration for Mistral3 models."""

    @override
    @classmethod
    def initialize(cls, pipeline_config: PipelineConfig) -> Self:
        """Initializes a MistralConfig instance from pipeline configuration.

        This method creates a config instance with all fields that can be determined
        from the pipeline configuration.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.

        Returns:
            An initialized MistralConfig instance.
        """
        huggingface_config = pipeline_config.model.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for '{pipeline_config.model.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )
        return cls.initialize_from_config(
            pipeline_config, huggingface_config.text_config
        )
