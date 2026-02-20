# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from max.config import (
    MAXConfig,
    convert_max_config_value,
    deep_merge_max_configs,
    get_default_max_config_file_section_name,
    resolve_max_config_inheritance,
)

from .bfloat16_utils import float32_to_bfloat16_as_uint16
from .config import AudioGenerationConfig, PipelineConfig
from .config_enums import (
    PipelineRole,
    RepoType,
    RopeType,
    SupportedEncoding,
    is_float4_encoding,
    parse_supported_encoding_from_file_name,
    supported_encoding_dtype,
    supported_encoding_quantization,
    supported_encoding_supported_devices,
    supported_encoding_supported_on,
)
from .embeddings_pipeline import EmbeddingsPipeline, EmbeddingsPipelineType
from .float8 import parse_float8_config
from .hf_utils import (
    HuggingFaceRepo,
    download_weight_files,
    generate_local_model_path,
    try_to_load_from_cache,
    validate_hf_repo_access,
)
from .interfaces import (
    AlwaysSignalBuffersMixin,
    KVCacheMixin,
    ModelInputs,
    ModelOutputs,
    PipelineModel,
)
from .kv_cache_config import KVCacheConfig
from .lora import LoRAManager
from .lora_config import LoRAConfig
from .lora_request_processor import LoRARequestProcessor
from .memory_estimation import MemoryEstimator
from .model_config import MAXModelConfig, MAXModelConfigBase
from .pipeline_variants import PixelGenerationPipeline, TextGenerationPipeline
from .pipeline_variants.overlap_text_generation import (
    OverlapTextGenerationPipeline,
)
from .pixel_tokenizer import PixelGenerationTokenizer
from .profiling_config import ProfilingConfig
from .registry import PIPELINE_REGISTRY, SupportedArchitecture
from .sampling import (
    SamplingConfig,
    rejection_sampler,
    rejection_sampler_with_residuals,
    token_sampler,
)
from .speculative_config import SpeculativeConfig
from .speculative_decoding import (
    EAGLESpeculativeDecodingPipeline,
    SpeculativeDecodingPipelineBase,
    StandaloneSpeculativeDecodingPipeline,
)
from .speech_token_pipeline import SpeechTokenGenerationPipeline
from .tokenizer import (
    IdentityPipelineTokenizer,
    PreTrainedPipelineTokenizer,
    TextAndVisionTokenizer,
    TextTokenizer,
    max_tokens_to_generate,
)
from .utils import CompilationTimer, upper_bounded_default
from .weight_path_parser import WeightPathParser

__all__ = [
    "PIPELINE_REGISTRY",
    "AlwaysSignalBuffersMixin",
    "AudioGenerationConfig",
    "CompilationTimer",
    "EAGLESpeculativeDecodingPipeline",
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
    "MemoryEstimator",
    "ModelInputs",
    "ModelOutputs",
    "OverlapTextGenerationPipeline",
    "PipelineConfig",
    "PipelineModel",
    "PipelineRole",
    "PixelGenerationPipeline",
    "PixelGenerationTokenizer",
    "PreTrainedPipelineTokenizer",
    "ProfilingConfig",
    "RepoType",
    "RopeType",
    "SamplingConfig",
    "SpeculativeConfig",
    "SpeculativeDecodingBase",
    "SpeechTokenGenerationPipeline",
    "StandaloneSpeculativeDecodingPipeline",
    "SupportedArchitecture",
    "SupportedEncoding",
    "TextAndVisionTokenizer",
    "TextGenerationPipeline",
    "TextTokenizer",
    "WeightPathParser",
    "convert_max_config_value",
    "deep_merge_max_configs",
    "download_weight_files",
    "float32_to_bfloat16_as_uint16",
    "generate_local_model_path",
    "get_default_max_config_file_section_name",
    "is_float4_encoding",
    "max_tokens_to_generate",
    "parse_float8_config",
    "parse_supported_encoding_from_file_name",
    "rejection_sampler",
    "rejection_sampler_with_residuals",
    "resolve_max_config_inheritance",
    "supported_encoding_dtype",
    "supported_encoding_quantization",
    "supported_encoding_supported_devices",
    "supported_encoding_supported_on",
    "token_sampler",
    "try_to_load_from_cache",
    "upper_bounded_default",
    "validate_hf_repo_access",
]
