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

from .architectures import register_all_models
from .core import (
    AudioGenerationResponse,
    EmbeddingsGenerator,
    EmbeddingsResponse,
    InputContext,
    LogProbabilities,
    PipelinesFactory,
    PipelineTask,
    SamplingParams,
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
from .lib.config import (
    AudioGenerationConfig,
    PipelineConfig,
    PrependPromptSpeechTokens,
    PrometheusMetricsMode,
)
from .lib.config_enums import (
    PipelineEngine,
    PipelineRole,
    RepoType,
    RopeType,
    SupportedEncoding,
)
from .lib.embeddings_pipeline import EmbeddingsPipeline
from .lib.hf_utils import (
    HuggingFaceFile,
    download_weight_files,
    repo_exists_with_retry,
)
from .lib.max_config import (
    KVCacheConfig,
    ProfilingConfig,
    SamplingConfig,
)
from .lib.memory_estimation import MEMORY_ESTIMATOR
from .lib.model_config import MAXModelConfig
from .lib.pipeline import (
    ModelInputs,
    ModelOutputs,
    PipelineModel,
    TextGenerationPipeline,
    upper_bounded_default,
)
from .lib.registry import PIPELINE_REGISTRY, SupportedArchitecture
from .lib.speculative_decoding import SpeculativeDecodingTextGenerationPipeline
from .lib.speech_token_pipeline import SpeechTokenGenerationPipeline
from .lib.tokenizer import (
    IdentityPipelineTokenizer,
    PipelineTokenizer,
    PreTrainedPipelineTokenizer,
    TextAndVisionTokenizer,
    TextTokenizer,
)

# Hydrate the registry.
register_all_models()

__all__ = [
    "AudioGenerationConfig",
    "download_weight_files",
    "AudioGenerationResponse",
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
    "PipelineRole",
    "PipelineModel",
    "PipelinesFactory",
    "PipelineTask",
    "PipelineTokenizer",
    "PreTrainedPipelineTokenizer",
    "PrependPromptSpeechTokens",
    "PrometheusMetricsMode",
    "ProfilingConfig",
    "repo_exists_with_retry",
    "RepoType",
    "RopeType",
    "SamplingConfig",
    "SamplingParams",
    "SpeculativeDecodingTextGenerationPipeline",
    "SpeechTokenGenerationPipeline",
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
