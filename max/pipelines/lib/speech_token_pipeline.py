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
"""Speech token generation pipeline for TTS model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypeVar

from max.graph.weights import WeightsAdapter, WeightsFormat
from max.pipelines.core import InputContext

from .pipeline import PipelineModel, TextGenerationPipeline

if TYPE_CHECKING:
    from .config import PipelineConfig

T = TypeVar("T", bound=InputContext)

logger = logging.getLogger("max.pipelines")


class SpeechTokenGenerationPipeline(TextGenerationPipeline):
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: type[PipelineModel],
        eos_token_id: int,
        weight_adapters: dict[WeightsFormat, WeightsAdapter],
    ) -> None:
        super().__init__(
            pipeline_config,
            pipeline_model,
            eos_token_id,
            weight_adapters,
        )
