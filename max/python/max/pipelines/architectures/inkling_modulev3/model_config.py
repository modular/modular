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
"""Deployment config for registered Inkling decoder and vision roles."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from max.pipelines.lib import MAXModelConfig, PipelineConfig
from max.pipelines.lib.graph_specs import GraphRole
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.registration_runtime import DeploymentConfig
from transformers import AutoConfig
from typing_extensions import Self

from ..inkling.model_config import InklingConfig as BaseInklingConfig
from ..inkling.processor import load_processor_config

_EXPERT_SCALE = re.compile(
    r"(\.experts\.w(?:13|2))_(weight_scale(?:_2)?|input_scale)$"
)


@dataclass(kw_only=True)
class InklingConfig(BaseInklingConfig, DeploymentConfig):
    """Finalizes both roles and records image preprocessing and outputs."""

    weight_prefixes: ClassVar[Mapping[str, str]] = {
        GraphRole.DECODER: "decoder.",
        GraphRole.VISION: "vision.",
    }
    outputs: ModelOutputConfig = field(default_factory=ModelOutputConfig)
    image_mean: tuple[float, ...] = ()
    image_std: tuple[float, ...] = ()

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Builds the shared config and reads its image normalization."""
        config = super().initialize(
            pipeline_config, model_config, max_seq_len=max_seq_len
        )
        model_config = model_config or pipeline_config.model
        processor = load_processor_config(
            model_config.model_path, model_config.huggingface_model_revision
        )["image_processor"]
        config.image_mean = tuple(processor["image_mean"])
        config.image_std = tuple(processor["image_std"])
        return config

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: Mapping[str, Any],
        outputs: ModelOutputConfig | None = None,
    ) -> None:
        """Parses expert quantization using the pre-module scale names."""
        language = {
            _EXPERT_SCALE.sub(
                r"\1_weight.\2", name.removeprefix("decoder.")
            ): value
            for name, value in state_dict.items()
            if name.startswith("decoder.")
        }
        super().finalize(huggingface_config, language)
        self.outputs = outputs or ModelOutputConfig()
