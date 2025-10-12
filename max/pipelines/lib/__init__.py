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
from .config_enums import PipelineRole, RepoType, RopeType, SupportedEncoding
from .embeddings_pipeline import EmbeddingsPipeline, EmbeddingsPipelineType
from .hf_utils import (
    HuggingFaceRepo,
    download_weight_files,
    generate_local_model_path,
    try_to_load_from_cache,
    validate_hf_repo_access,
)
from .kv_cache_config import KVCacheConfig
from .lora import LoRAManager
from .lora_config import LoRAConfig
from .lora_request_processor import LoRARequestProcessor
from .max_config import (
    MAXConfig,
    convert_max_config_value,
    deep_merge_max_configs,
    get_default_max_config_file_section_name,
    resolve_max_config_inheritance,
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
from .profiling_config import ProfilingConfig
from .ragged_token_merger import ragged_token_merger
from .registry import PIPELINE_REGISTRY, SupportedArchitecture
from .sampling import (
    SamplingConfig,
    rejection_sampler,
    rejection_sampler_with_residuals,
    token_sampler,
)
from .speculative_decoding import SpeculativeDecodingTextGenerationPipeline
from .speech_token_pipeline import SpeechTokenGenerationPipeline
from .tokenizer import (
    IdentityPipelineTokenizer,
    PreTrainedPipelineTokenizer,
    TextAndVisionTokenizer,
    TextTokenizer,
    max_tokens_to_generate,
)
from .weight_path_parser import WeightPathParser

__all__ = [
    "MEMORY_ESTIMATOR",
    "PIPELINE_REGISTRY",
    "AudioGenerationConfig",
    "EmbeddingsPipeline",
    "EmbeddingsPipelineType",
    "HuggingFaceRepo",
    "IdentityPipelineTokenizer",
    "KVCacheConfig",
    "KVCacheMixin",
    "LoRAConfig",
    "LoRAManager",
    "LoRARequestProcessor",
    "MAXConfig",
    "MAXModelConfig",
    "MAXModelConfigBase",
    "ModelInputs",
    "ModelOutputs",
    "PipelineConfig",
    "PipelineModel",
    "PipelineRole",
    "PreTrainedPipelineTokenizer",
    "ProfilingConfig",
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
    "WeightPathParser",
    "convert_max_config_value",
    "deep_merge_max_configs",
    "download_weight_files",
    "generate_local_model_path",
    "get_default_max_config_file_section_name",
    "get_paged_manager",
    "max_tokens_to_generate",
    "ragged_token_merger",
    "rejection_sampler",
    "rejection_sampler_with_residuals",
    "resolve_max_config_inheritance",
    "token_sampler",
    "try_to_load_from_cache",
    "upper_bounded_default",
    "validate_hf_repo_access",
]
