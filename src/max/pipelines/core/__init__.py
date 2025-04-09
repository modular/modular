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

from typing import Callable as _Callable
from typing import Union as _Union

from .context import InputContext, TextAndVisionContext, TextContext
from .interfaces import (
    EmbeddingsGenerator,
    EmbeddingsResponse,
    LogProbabilities,
    PipelineTask,
    PipelineTokenizer,
    TextGenerationResponse,
    TextGenerationStatus,
    TextResponse,
    TokenGenerator,
    TokenGeneratorContext,
    TokenGeneratorRequest,
    TokenGeneratorRequestFunction,
    TokenGeneratorRequestMessage,
    TokenGeneratorRequestTool,
    TokenGeneratorResponseFormat,
)

PipelinesFactory = _Callable[[], _Union[TokenGenerator, EmbeddingsGenerator]]

__all__ = [
    "InputContext",
    "TextAndVisionContext",
    "TextContext",
    "EmbeddingsGenerator",
    "LogProbabilities",
    "TextResponse",
    "EmbeddingsResponse",
    "TextGenerationStatus",
    "TextGenerationResponse",
    "PipelineTask",
    "PipelineTokenizer",
    "TokenGenerator",
    "TokenGeneratorContext",
    "TokenGeneratorRequest",
    "TokenGeneratorRequestFunction",
    "TokenGeneratorRequestMessage",
    "TokenGeneratorRequestTool",
    "TokenGeneratorResponseFormat",
    "PipelinesFactory",
]
