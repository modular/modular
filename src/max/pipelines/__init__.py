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

from .config import PipelineConfig
from .config_enums import PipelineEngine, RepoType, RopeType, SupportedEncoding
from .core import (
    EmbeddingsGenerator,
    EmbeddingsResponse,
    InputContext,
    LogProbabilities,
    PipelinesFactory,
    PipelineTask,
    TextAndVisionContext,
    TextContext,
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
from .embeddings_pipeline import EmbeddingsPipeline
from .hf_utils import (
    HuggingFaceFile,
    download_weight_files,
    repo_exists_with_retry,
)
from .log_probabilities import compute_log_probabilities
from .max_config import (
    KVCacheConfig,
    ProfilingConfig,
    SamplingConfig,
)
from .memory_estimation import MEMORY_ESTIMATOR
from .model_config import MAXModelConfig
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
    PipelineTokenizer,
    PreTrainedPipelineTokenizer,
    TextAndVisionTokenizer,
    TextTokenizer,
)

__all__ = [
    "compute_log_probabilities",
    "download_weight_files",
    "EmbeddingsGenerator",
    "EmbeddingsPipeline",
    "EmbeddingsResponse",
    "HuggingFaceFile",
    "IdentityPipelineTokenizer",
    "InputContext",
    "KVCacheConfig",
    "LogProbabilities",
    "MAXModelConfig",
    "MEMORY_ESTIMATOR",
    "ModelInputs",
    "ModelOutputs",
    "PIPELINE_REGISTRY",
    "PipelineConfig",
    "PipelineEngine",
    "PipelineModel",
    "PipelinesFactory",
    "PipelineTask",
    "PipelineTokenizer",
    "PreTrainedPipelineTokenizer",
    "ProfilingConfig",
    "repo_exists_with_retry",
    "RepoType",
    "RopeType",
    "SamplingConfig",
    "SpeculativeDecodingTextGenerationPipeline",
    "SupportedArchitecture",
    "SupportedEncoding",
    "TextAndVisionContext",
    "TextAndVisionTokenizer",
    "TextContext",
    "TextGenerationPipeline",
    "TextGenerationResponse",
    "TextGenerationStatus",
    "TextResponse",
    "TextTokenizer",
    "TokenGenerator",
    "TokenGeneratorContext",
    "TokenGeneratorRequest",
    "TokenGeneratorRequestFunction",
    "TokenGeneratorRequestMessage",
    "TokenGeneratorRequestTool",
    "TokenGeneratorResponseFormat",
    "upper_bounded_default",
]
