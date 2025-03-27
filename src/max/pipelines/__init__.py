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

from .config import PipelineConfig
from .config_enums import PipelineEngine, RepoType, RopeType, SupportedEncoding
from .embeddings_pipeline import EmbeddingsPipeline
from .hf_utils import HuggingFaceFile, download_weight_files
from .interfaces import (
    EmbeddingsGenerator,
    EmbeddingsResponse,
    LogProbabilities,
    PipelineTask,
    PipelineTokenizer,
    TextResponse,
    TokenGenerator,
    TokenGeneratorContext,
    TokenGeneratorRequest,
    TokenGeneratorRequestFunction,
    TokenGeneratorRequestMessage,
    TokenGeneratorRequestTool,
    TokenGeneratorResponseFormat,
)
from .log_probabilities import compute_log_probabilities
from .max_config import (
    KVCacheConfig,
    MAXModelConfig,
    ProfilingConfig,
    SamplingConfig,
)
from .pipeline import (
    ModelInputs,
    ModelOutputs,
    PipelineModel,
    TextGenerationPipeline,
    upper_bounded_default,
)
from .registry import PIPELINE_REGISTRY, SupportedArchitecture
from .speculative_decoding import SpeculativeDecodingTextGenerationPipeline
from .tokenizer import (
    IdentityPipelineTokenizer,
    PreTrainedPipelineTokenizer,
    TextAndVisionTokenizer,
    TextTokenizer,
)

PipelinesFactory = _Callable[[], _Union[TokenGenerator, EmbeddingsGenerator]]


__all__ = [
    "LogProbabilities",
    "HuggingFaceFile",
    "PipelineConfig",
    "ProfilingConfig",
    "KVCacheConfig",
    "MAXModelConfig",
    "PipelineEngine",
    "PipelineTask",
    "PIPELINE_REGISTRY",
    "SamplingConfig",
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
    "PipelineTokenizer",
    "PreTrainedPipelineTokenizer",
    "TextTokenizer",
    "TextAndVisionTokenizer",
    "TextGenerationPipeline",
    "RepoType",
    "RopeType",
    "PipelineModel",
    "ModelInputs",
    "ModelOutputs",
    "TextResponse",
    "EmbeddingsGenerator",
    "EmbeddingsPipeline",
    "EmbeddingsResponse",
    "SpeculativeDecodingTextGenerationPipeline",
    "compute_log_probabilities",
    "upper_bounded_default",
    "download_weight_files",
]
