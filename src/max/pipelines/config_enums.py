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

"""Standardized configuration for Pipeline Inference."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from max.driver import (
    DeviceSpec,
)
from max.dtype import DType
from max.graph.quantization import QuantizationEncoding


class RepoType(str, Enum):
    """Specifies the source location type of a model repository.

    This determines how model configuration and weight files are located and loaded.
    """

    online = "online"
    """Indicates an online repository, typically hosted on HuggingFace Hub.

    Paths for weights within an online repository are resolved primarily
    through the local HuggingFace cache, but fall back to downloading through
    the HF API if the local cache isn't populated.
    """

    local = "local"
    """Indicates a repository stored on the local filesystem.

    Paths for weights within a local repository are resolved first directly
    (as absolute paths or relative to the current working directory), and
    then relative to the repository's root directory.
    """


# Reference: https://github.com/ggerganov/llama.cpp/blob/eb5c3dc64bd967f2e23c87d9dec195f45468de60/src/llama.cpp#L20778
class RopeType(str, Enum):
    none = "none"
    normal = "normal"
    neox = "neox"


class PipelineEngine(str, Enum):
    MAX = "max"
    HUGGINGFACE = "huggingface"


class SupportedEncoding(str, Enum):
    """All possible encodings which may be supported by a particular model."""

    float32 = "float32"
    bfloat16 = "bfloat16"
    q4_k = "q4_k"
    q4_0 = "q4_0"
    q6_k = "q6_k"
    gptq = "gptq"

    @classmethod
    def parse_from_file_name(cls, name: str):
        # TODO(AITLIB-127): Robustify detection of quantization encoding
        name = name.lower()
        if "f32" in name or "fp32" in name or "float32" in name:
            return SupportedEncoding.float32
        elif "bf16" in name or "bfloat16" in name:
            return SupportedEncoding.bfloat16
        elif "q4_k_m" in name:
            return SupportedEncoding.q4_k
        elif "q4_0" in name:
            return SupportedEncoding.q4_0
        elif "q6_k" in name:
            return SupportedEncoding.q6_k
        elif "gptq" in name:
            return SupportedEncoding.gptq
        else:
            return None

    @property
    def quantization_encoding(self) -> Optional[QuantizationEncoding]:
        if self not in _SUPPORTED_ENCODING_TO_QUANTIZATION_ENCODING:
            raise ValueError(
                f"SupportedEncoding({self}) does not have corresponding QuantizationEncoding."
            )
        return _SUPPORTED_ENCODING_TO_QUANTIZATION_ENCODING[self]

    @property
    def dtype(self) -> DType:
        """The underlying model dtype associated with a quantization_encoding."""
        if self not in _SUPPORTED_ENCODING_TO_DTYPE:
            raise ValueError(
                f"SupportedEncoding({self}) does not have corresponding dtype."
            )
        return _SUPPORTED_ENCODING_TO_DTYPE[self]

    @property
    def cache_dtype(self) -> DType:
        """The underlying dtype used in the kvcache for correctness."""
        if self not in _SUPPORTED_ENCODING_TO_CACHE_DTYPE:
            raise ValueError(
                f"SupportedEncoding({self}) does not have corresponding cache dtype."
            )

        return _SUPPORTED_ENCODING_TO_CACHE_DTYPE[self]

    def supported_on(self, device_spec: DeviceSpec) -> bool:
        """Returns whether this quantization encoding is supported on a device."""
        return device_spec.device_type in _SUPPORTED_DEVICES[self]


_SUPPORTED_ENCODING_TO_DTYPE = {
    SupportedEncoding.float32: DType.float32,
    SupportedEncoding.bfloat16: DType.bfloat16,
    SupportedEncoding.q4_k: DType.uint8,
    SupportedEncoding.q4_0: DType.uint8,
    SupportedEncoding.q6_k: DType.uint8,
    SupportedEncoding.gptq: DType.uint8,
}


_SUPPORTED_ENCODING_TO_CACHE_DTYPE = {
    SupportedEncoding.float32: DType.float32,
    SupportedEncoding.bfloat16: DType.bfloat16,
    SupportedEncoding.q4_k: DType.float32,
    SupportedEncoding.q4_0: DType.float32,
    SupportedEncoding.q6_k: DType.float32,
    SupportedEncoding.gptq: DType.bfloat16,
}

_SUPPORTED_ENCODING_TO_QUANTIZATION_ENCODING = {
    SupportedEncoding.float32: None,
    SupportedEncoding.bfloat16: None,
    SupportedEncoding.q4_k: QuantizationEncoding.Q4_K,
    SupportedEncoding.q4_0: QuantizationEncoding.Q4_0,
    SupportedEncoding.q6_k: QuantizationEncoding.Q6_K,
    SupportedEncoding.gptq: QuantizationEncoding.GPTQ,
}


# Basic validation for supported devices for each type of encoding.
_SUPPORTED_DEVICES: dict[SupportedEncoding, tuple[str, ...]] = {
    SupportedEncoding.float32: ("cpu", "gpu"),
    SupportedEncoding.bfloat16: ("gpu",),
    SupportedEncoding.q4_k: ("cpu",),
    SupportedEncoding.q4_0: ("cpu",),
    SupportedEncoding.q6_k: ("cpu",),
    SupportedEncoding.gptq: ("gpu",),
}

# Store a map of checkpoint encodings that can be cast to another dtype while
# keeping similar results. Maps the requested encoding to an acceptable
# alternate checkpoint encoding.
_ALTERNATE_ENCODINGS = {
    SupportedEncoding.float32: SupportedEncoding.bfloat16,
    SupportedEncoding.bfloat16: SupportedEncoding.float32,
}
