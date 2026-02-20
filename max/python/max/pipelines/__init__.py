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

"""Types to interface with ML pipelines such as text/token/pixel generation."""

from .architectures import register_all_models
from .core import PixelContext, TextAndVisionContext, TextContext, TTSContext
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
    is_float4_encoding,
    parse_supported_encoding_from_file_name,
    supported_encoding_dtype,
    supported_encoding_quantization,
    supported_encoding_supported_devices,
    supported_encoding_supported_on,
)
from .lib.embeddings_pipeline import EmbeddingsPipeline, EmbeddingsPipelineType
from .lib.hf_utils import download_weight_files
from .lib.interfaces import (
    GenerateMixin,
    ModelInputs,
    ModelOutputs,
    PipelineModel,
)
from .lib.kv_cache_config import KVCacheConfig
from .lib.lora import ADAPTER_CONFIG_FILE
from .lib.memory_estimation import MemoryEstimator
from .lib.model_config import MAXModelConfig
from .lib.pipeline_variants.pixel_generation import PixelGenerationPipeline
from .lib.pipeline_variants.text_generation import (
    TextGenerationPipeline,
    TextGenerationPipelineInterface,
)
from .lib.profiling_config import ProfilingConfig
from .lib.registry import PIPELINE_REGISTRY, SupportedArchitecture
from .lib.sampling.sampling_config import SamplingConfig
from .lib.speech_token_pipeline import SpeechTokenGenerationPipeline
from .lib.tokenizer import (
    IdentityPipelineTokenizer,
    PreTrainedPipelineTokenizer,
    TextAndVisionTokenizer,
    TextTokenizer,
)
from .lib.utils import upper_bounded_default

# Hydrate the registry.
register_all_models()

__all__ = [
    "ADAPTER_CONFIG_FILE",
    "PIPELINE_REGISTRY",
    "AudioGenerationConfig",
    "EmbeddingsPipeline",
    "EmbeddingsPipelineType",
    "GenerateMixin",
    "IdentityPipelineTokenizer",
    "KVCacheConfig",
    "MAXModelConfig",
    "MemoryEstimator",
    "ModelInputs",
    "ModelOutputs",
    "PipelineConfig",
    "PipelineModel",
    "PipelineRole",
    "PixelContext",
    "PixelGenerationPipeline",
    "PreTrainedPipelineTokenizer",
    "PrependPromptSpeechTokens",
    "ProfilingConfig",
    "PrometheusMetricsMode",
    "RepoType",
    "RopeType",
    "SamplingConfig",
    "SpeechTokenGenerationPipeline",
    "SupportedArchitecture",
    "SupportedEncoding",
    "TTSContext",
    "TextAndVisionContext",
    "TextAndVisionTokenizer",
    "TextContext",
    "TextGenerationPipeline",
    "TextGenerationPipelineInterface",
    "TextTokenizer",
    "download_weight_files",
    "is_float4_encoding",
    "parse_supported_encoding_from_file_name",
    "supported_encoding_dtype",
    "supported_encoding_quantization",
    "supported_encoding_supported_devices",
    "supported_encoding_supported_on",
    "upper_bounded_default",
]
