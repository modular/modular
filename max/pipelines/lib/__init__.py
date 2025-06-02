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

from .config import AudioGenerationConfig, PipelineConfig
from .config_enums import (
    PipelineEngine,
    PipelineRole,
    RepoType,
    RopeType,
    SupportedEncoding,
)
from .embeddings_pipeline import EmbeddingsPipeline
from .hf_utils import (
    HuggingFaceFile,
    HuggingFaceRepo,
    download_weight_files,
    generate_local_model_path,
    repo_exists_with_retry,
)
from .max_config import (
    KVCacheConfig,
    ProfilingConfig,
    SamplingConfig,
)
from .memory_estimation import MEMORY_ESTIMATOR
from .model_config import MAXModelConfig, MAXModelConfigBase
from .pipeline import (
    KVCacheMixin,
    ModelInputs,
    ModelOutputs,
    PipelineModel,
    TextGenerationPipeline,
    get_paged_manager,
    upper_bounded_default,
)
from .ragged_token_merger import ragged_token_merger
from .registry import PIPELINE_REGISTRY, SupportedArchitecture
from .sampling import rejection_sampler, token_sampler
from .speculative_decoding import SpeculativeDecodingTextGenerationPipeline
from .speech_token_pipeline import SpeechTokenGenerationPipeline
from .tokenizer import (
    IdentityPipelineTokenizer,
    PipelineTokenizer,
    PreTrainedPipelineTokenizer,
    TextAndVisionTokenizer,
    TextTokenizer,
)

__all__ = [
    "AudioGenerationConfig",
    "download_weight_files",
    "EmbeddingsPipeline",
    "generate_local_model_path",
    "HuggingFaceFile",
    "HuggingFaceRepo",
    "IdentityPipelineTokenizer",
    "KVCacheConfig",
    "KVCacheMixin",
    "MAXModelConfig",
    "MAXModelConfigBase",
    "MEMORY_ESTIMATOR",
    "ModelInputs",
    "ModelOutputs",
    "PipelineConfig",
    "PipelineEngine",
    "PipelineModel",
    "PipelineRole",
    "PIPELINE_REGISTRY",
    "PipelineTokenizer",
    "PreTrainedPipelineTokenizer",
    "ProfilingConfig",
    "ragged_token_merger",
    "rejection_sampler",
    "repo_exists_with_retry",
    "RepoType",
    "RopeType",
    "SamplingConfig",
    "SpeculativeDecodingTextGenerationPipeline",
    "SpeechTokenGenerationPipeline",
    "SupportedArchitecture",
    "SupportedEncoding",
    "TextAndVisionTokenizer",
    "TextGenerationPipeline",
    "TextTokenizer",
    "token_sampler",
    "upper_bounded_default",
    "get_paged_manager",
]
