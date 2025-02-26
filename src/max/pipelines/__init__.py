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

"""Types to interface with ML pipelines such as text/token generation."""

from typing import Callable as _Callable
from typing import Union as _Union

from .config import (
    PipelineConfig,
    PipelineEngine,
    RopeType,
    SamplingParams,
    SupportedEncoding,
    WeightsFormat,
)
from .context import InputContext, TextAndVisionContext, TextContext
from .embeddings_pipeline import EmbeddingsPipeline
from .hf_utils import HuggingFaceFile
from .interfaces import (
    EmbeddingsGenerator,
    PipelineTask,
    PipelineTokenizer,
    TokenGenerator,
    TokenGeneratorContext,
    TokenGeneratorRequest,
    TokenGeneratorRequestFunction,
    TokenGeneratorRequestMessage,
    TokenGeneratorRequestTool,
    TokenGeneratorResponseFormat,
)
from .pipeline import (
    ModelInputs,
    ModelOutputs,
    PipelineModel,
    TextGenerationPipeline,
    upper_bounded_default,
)
from .registry import PIPELINE_REGISTRY, SupportedArchitecture
from .response import EmbeddingsResponse, LogProbabilities, TextResponse
from .tokenizer import (
    IdentityPipelineTokenizer,
    PreTrainedPipelineTokenizer,
    TextAndVisionTokenizer,
    TextTokenizer,
)

PipelinesFactory = _Callable[[], _Union[TokenGenerator, EmbeddingsGenerator]]


__all__ = [
    "HuggingFaceFile",
    "PipelineConfig",
    "PipelineEngine",
    "PipelineTask",
    "PIPELINE_REGISTRY",
    "SamplingParams",
    "SupportedArchitecture",
    "SupportedEncoding",
    "TokenGenerator",
    "TokenGeneratorContext",
    "TokenGeneratorRequest",
    "TokenGeneratorRequestMessage",
    "TokenGeneratorResponseFormat",
    "TokenGeneratorRequestTool",
    "TokenGeneratorResponseFormat",
    "TokenGeneratorRequestFunction",
    "IdentityPipelineTokenizer",
    "InputContext",
    "TextContext",
    "TextAndVisionContext",
    "PipelineTokenizer",
    "PreTrainedPipelineTokenizer",
    "TextTokenizer",
    "TextAndVisionTokenizer",
    "TextGenerationPipeline",
    "WeightsFormat",
    "RopeType",
    "PipelineModel",
    "ModelInputs",
    "ModelOutputs",
    "TextResponse",
    "LogProbabilities",
    "EmbeddingsGenerator",
    "EmbeddingsPipeline",
    "EmbeddingsResponse",
    "upper_bounded_default",
]
