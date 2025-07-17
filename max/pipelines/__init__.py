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
    EmbeddingsGenerator,
    PipelinesFactory,
    TextAndVisionContext,
    TextContext,
    TokenGeneratorContext,
    TokenGeneratorRequest,
    TokenGeneratorRequestFunction,
    TokenGeneratorRequestMessage,
    TokenGeneratorRequestTool,
    TokenGeneratorResponseFormat,
    TTSContext,
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
from .lib.hf_utils import download_weight_files
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
    "EmbeddingsGenerator",
    "EmbeddingsPipeline",
    "IdentityPipelineTokenizer",
    "KVCacheConfig",
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
    "PipelineTokenizer",
    "PreTrainedPipelineTokenizer",
    "PrependPromptSpeechTokens",
    "PrometheusMetricsMode",
    "ProfilingConfig",
    "RepoType",
    "RopeType",
    "SamplingConfig",
    "SpeculativeDecodingTextGenerationPipeline",
    "SpeechTokenGenerationPipeline",
    "SupportedArchitecture",
    "SupportedEncoding",
    "TextAndVisionContext",
    "TextAndVisionTokenizer",
    "TextContext",
    "TextGenerationPipeline",
    "TextTokenizer",
    "TokenGeneratorContext",
    "TokenGeneratorRequest",
    "TokenGeneratorRequestFunction",
    "TokenGeneratorRequestMessage",
    "TokenGeneratorRequestTool",
    "TokenGeneratorResponseFormat",
    "upper_bounded_default",
    "TTSContext",
]
