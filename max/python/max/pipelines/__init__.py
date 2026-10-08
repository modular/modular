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

"""Interfaces for running models end to end: registry, configuration, and
pipeline classes for text, embeddings, and pixel generation.

A pipeline wires a model family into an executable workflow. A
:class:`~max.pipelines.lib.registry.SupportedArchitecture` registers the model in the
pipeline registry, :class:`~max.pipelines.lib.config.PipelineConfig` sets how it runs, and
pipeline classes such as :class:`TextGenerationPipeline`,
:class:`EmbeddingsPipeline`, and :class:`PixelGenerationPipeline` drive
tokenization, weight loading, and generation.
"""

from max.experimental.validation import EagerUsageValidator
from max.pipelines.weights.hf_utils import download_weight_files

from .architectures import register_all_models
from .diffusion.pipeline import PixelGenerationPipeline
from .lib.config import (
    KVCacheConfig,
    LoRAConfig,
    MAXModelConfig,
    PipelineArgs,
    PipelineConfig,
    PipelineRole,
    ProfilingConfig,
    PrometheusMetricsMode,
    RepoType,
    RopeType,
    SpeculativeConfig,
    SupportedEncoding,
    is_float4_encoding,
    parse_supported_encoding_from_file_name,
    supported_encoding_dtype,
    supported_encoding_quantization,
    supported_encoding_supported_devices,
    supported_encoding_supported_on,
)
from .lib.embeddings_pipeline import EmbeddingsPipeline, EmbeddingsPipelineType
from .lib.interfaces import (
    GenerateMixin,
    ModelInputs,
    ModelOutputs,
    PipelineModel,
)
from .lib.memory_estimation import MemoryEstimator
from .lib.pipeline_runtime_config import (
    EagerValidatorMode,
    PipelineRuntimeConfig,
)
from .lib.pipeline_variants.text_generation import (
    TextGenerationPipeline,
    TextGenerationPipelineInterface,
)
from .lib.registry import PIPELINE_REGISTRY, SupportedArchitecture
from .lib.tokenizer import (
    IdentityPipelineTokenizer,
    TextAndVisionTokenizer,
    TextTokenizer,
)
from .lib.utils import upper_bounded_default
from .lora import ADAPTER_CONFIG_FILE
from .modeling.eager_validation import eager_validator
from .sampling import SamplingConfig

# Hydrate the registry.
register_all_models()

__all__ = [
    "ADAPTER_CONFIG_FILE",
    "PIPELINE_REGISTRY",
    "EagerUsageValidator",
    "EagerValidatorMode",
    "EmbeddingsPipeline",
    "EmbeddingsPipelineType",
    "GenerateMixin",
    "IdentityPipelineTokenizer",
    "KVCacheConfig",
    "LoRAConfig",
    "MAXModelConfig",
    "MemoryEstimator",
    "ModelInputs",
    "ModelOutputs",
    "PipelineArgs",
    "PipelineConfig",
    "PipelineModel",
    "PipelineRole",
    "PipelineRuntimeConfig",
    "PixelGenerationPipeline",
    "ProfilingConfig",
    "PrometheusMetricsMode",
    "RepoType",
    "RopeType",
    "SamplingConfig",
    "SpeculativeConfig",
    "SupportedArchitecture",
    "SupportedEncoding",
    "TextAndVisionTokenizer",
    "TextGenerationPipeline",
    "TextGenerationPipelineInterface",
    "TextTokenizer",
    "download_weight_files",
    "eager_validator",
    "is_float4_encoding",
    "parse_supported_encoding_from_file_name",
    "supported_encoding_dtype",
    "supported_encoding_quantization",
    "supported_encoding_supported_devices",
    "supported_encoding_supported_on",
    "upper_bounded_default",
]
