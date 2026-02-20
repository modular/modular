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
"""Speculative decoding pipelines for MAX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from max.graph.weights import WeightsAdapter, WeightsFormat
from max.interfaces import (
    Pipeline,
    PipelineTokenizer,
    TextGenerationInputs,
    TextGenerationOutput,
    TextGenerationRequest,
)
from max.pipelines.core import TextContext

from ..speculative_config import SpeculativeMethod
from .base import SpeculativeDecodingPipelineBase
from .eagle import EAGLESpeculativeDecodingPipeline
from .ragged_token_merger import RaggedTokenMerger, ragged_token_merger
from .standalone import StandaloneSpeculativeDecodingPipeline

if TYPE_CHECKING:
    from ..config import PipelineConfig
    from ..interfaces import PipelineModel

__all__ = [
    "EAGLESpeculativeDecodingPipeline",
    "RaggedTokenMerger",
    "SpeculativeDecodingPipelineBase",
    "StandaloneSpeculativeDecodingPipeline",
    "ragged_token_merger",
]
