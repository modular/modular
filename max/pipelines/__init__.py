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
from .core import TextAndVisionContext, TextContext, TTSContext
from .lib.config import (
    AudioGenerationConfig,
    PipelineConfig,
    PrependPromptSpeechTokens,
    PrometheusMetricsMode,
)
from .lib.config_enums import (
    PipelineRole,
    RepoType,
    RopeType,
    SupportedEncoding,
)
from .lib.embeddings_pipeline import EmbeddingsPipeline, EmbeddingsPipelineType
from .lib.hf_utils import download_weight_files
from .lib.kv_cache_config import KVCacheConfig
from .lib.memory_estimation import MEMORY_ESTIMATOR
from .lib.model_config import MAXModelConfig
from .lib.pipeline import (
    GenerateMixin,
    ModelInputs,
    ModelOutputs,
    PipelineModel,
    TextGenerationPipeline,
    upper_bounded_default,
)
from .lib.profiling_config import ProfilingConfig
from .lib.registry import PIPELINE_REGISTRY, SupportedArchitecture
from .lib.sampling.sampling_config import SamplingConfig
from .lib.speculative_decoding import SpeculativeDecodingTextGenerationPipeline
from .lib.speech_token_pipeline import SpeechTokenGenerationPipeline
from .lib.tokenizer import (
    IdentityPipelineTokenizer,
    PreTrainedPipelineTokenizer,
    TextAndVisionTokenizer,
    TextTokenizer,
)

# Hydrate the registry.
register_all_models()

__all__ = [
    "MEMORY_ESTIMATOR",
    "PIPELINE_REGISTRY",
    "AudioGenerationConfig",
    "EmbeddingsPipeline",
    "EmbeddingsPipelineType",
    "GenerateMixin",
    "IdentityPipelineTokenizer",
    "KVCacheConfig",
    "MAXModelConfig",
    "ModelInputs",
    "ModelOutputs",
    "PipelineConfig",
    "PipelineModel",
    "PipelineRole",
    "PreTrainedPipelineTokenizer",
    "PrependPromptSpeechTokens",
    "ProfilingConfig",
    "PrometheusMetricsMode",
    "RepoType",
    "RopeType",
    "SamplingConfig",
    "SpeculativeDecodingTextGenerationPipeline",
    "SpeechTokenGenerationPipeline",
    "SupportedArchitecture",
    "SupportedEncoding",
    "TTSContext",
    "TextAndVisionContext",
    "TextAndVisionTokenizer",
    "TextContext",
    "TextGenerationPipeline",
    "TextTokenizer",
    "download_weight_files",
    "upper_bounded_default",
]
