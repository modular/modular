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

"""Configuration classes for MAX pipelines."""

from .config import (
    DEFAULT_MAX_BATCH_INPUT_TOKENS,
    AudioGenerationConfig,
    PipelineConfig,
    PrependPromptSpeechTokens,
    PrometheusMetricsMode,
    _format_config_entries,
)
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
from .kv_cache_config import KVCacheConfig
from .lora_config import LoRAConfig
from .model_config import MAXModelConfig, MAXModelConfigBase
from .profiling_config import ProfilingConfig
from .speculative_config import SpeculativeConfig, SpeculativeMethod

__all__ = [
    "DEFAULT_MAX_BATCH_INPUT_TOKENS",
    "AudioGenerationConfig",
    "KVCacheConfig",
    "LoRAConfig",
    "MAXModelConfig",
    "MAXModelConfigBase",
    "PipelineConfig",
    "PipelineRole",
    "PrependPromptSpeechTokens",
    "ProfilingConfig",
    "PrometheusMetricsMode",
    "RepoType",
    "RopeType",
    "SpeculativeConfig",
    "SpeculativeMethod",
    "SupportedEncoding",
    "_format_config_entries",
    "is_float4_encoding",
    "parse_supported_encoding_from_file_name",
    "supported_encoding_dtype",
    "supported_encoding_quantization",
    "supported_encoding_supported_devices",
    "supported_encoding_supported_on",
]
