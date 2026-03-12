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
"""Unified EAGLE pipeline: single fused graph for target + draft."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, final
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
from max.graph.weights import WeightsAdapter, WeightsFormat
from max.interfaces import (
    PipelineTokenizer,
    RequestID,
    TextGenerationInputs,
    TextGenerationOutput,
    TextGenerationRequest,
)
from max.kv_cache import PagedKVCacheManager
from max.pipelines.core import TextContext
from max.profiler import traced

from ..interfaces import PipelineModel
from ..pipeline_variants.text_generation import TextGenerationPipelineInterface

if TYPE_CHECKING:
    from ..config import PipelineConfig

logger = logging.getLogger("max.pipelines")


@final
class UnifiedEAGLEPipeline(TextGenerationPipelineInterface[TextContext]):
    """Pipeline for unified EAGLE: single fused graph handles target + draft.

    Unlike EAGLESpeculativeDecodingPipeline which manages two separate models,
    this pipeline uses a single model that runs both target forward and draft
    generation in one compiled graph call. Rejection sampling also happens
    in-graph (greedy acceptance).
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: type[PipelineModel[TextContext]],
        eos_token_id: int,
        weight_adapters: dict[WeightsFormat, WeightsAdapter],
        tokenizer: PipelineTokenizer[
            TextContext,
            npt.NDArray[np.integer[Any]],
            TextGenerationRequest,
        ],
        draft_pipeline_model: type[PipelineModel[TextContext]] | None = None,
        draft_weight_adapters: dict[WeightsFormat, WeightsAdapter]
        | None = None,
    ) -> None:
        self._pipeline_config = pipeline_config
        self._tokenizer = tokenizer
        # TODO: implement this properly
        self._kv_manager = MagicMock()
        logger.debug("Hello from UnifiedEAGLEPipeline")

    @property
    def pipeline_config(self) -> PipelineConfig:
        """Returns the pipeline configuration."""
        return self._pipeline_config

    @property
    def tokenizer(
        self,
    ) -> PipelineTokenizer[
        TextContext,
        npt.NDArray[np.integer[Any]],
        TextGenerationRequest,
    ]:
        """Returns the tokenizer for this pipeline."""
        return self._tokenizer

    @property
    def kv_manager(self) -> PagedKVCacheManager:
        """Returns the KV cache manager for this pipeline."""
        # TODO: implement this properly
        return self._kv_manager

    @traced
    def execute(
        self,
        inputs: TextGenerationInputs[TextContext],
    ) -> dict[RequestID, TextGenerationOutput]:
        """Executes Unified EAGLE speculative decoding pipeline."""
        raise NotImplementedError()

    def release(self, request_id: RequestID) -> None:
        """Releases the resources associated with the request."""
        pass
