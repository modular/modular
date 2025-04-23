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
from .embeddings_pipeline import EmbeddingsPipeline
from .hf_utils import (
    HuggingFaceFile,
    HuggingFaceRepo,
    download_weight_files,
    repo_exists_with_retry,
)
from .log_probabilities import compute_log_probabilities, log_softmax
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
    upper_bounded_default,
)
from .registry import PIPELINE_REGISTRY, SupportedArchitecture
from .sampling import rejection_sampler, token_sampler
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
    "EmbeddingsPipeline",
    "HuggingFaceFile",
    "HuggingFaceRepo",
    "IdentityPipelineTokenizer",
    "KVCacheConfig",
    "KVCacheMixin",
    "log_softmax",
    "MAXModelConfig",
    "MAXModelConfigBase",
    "MEMORY_ESTIMATOR",
    "ModelInputs",
    "ModelOutputs",
    "PipelineConfig",
    "PipelineEngine",
    "PipelineModel",
    "PIPELINE_REGISTRY",
    "PipelineTokenizer",
    "PreTrainedPipelineTokenizer",
    "ProfilingConfig",
    "rejection_sampler",
    "repo_exists_with_retry",
    "RepoType",
    "RopeType",
    "SamplingConfig",
    "SpeculativeDecodingTextGenerationPipeline",
    "SupportedArchitecture",
    "SupportedEncoding",
    "TextAndVisionTokenizer",
    "TextGenerationPipeline",
    "TextTokenizer",
    "token_sampler",
    "upper_bounded_default",
]
