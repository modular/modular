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
"""Speculative decoding pipelines and configuration for MAX."""

from .config import (
    MAGIC_DRAFT_TOKEN_ID,
    RejectionSamplingStrategy,
    SpeculativeConfig,
    SpeculativeMethod,
    VerifyWidthRange,
)
from .draft_weights import (
    NO_ALIASES,
    DraftAliases,
    validate_draft_state_dict,
)
from .ragged_token_merger import RaggedTokenMerger, ragged_token_merger
from .spec_input_types import (
    SpecDecodeGraphInputs,
    SpecDecodeInputTypeSpec,
    SpecDecodeTailValues,
    build_spec_decode_input_types,
    decode_spec_decode_input_values,
    decode_spec_decode_tail,
)

__all__ = [
    "MAGIC_DRAFT_TOKEN_ID",
    "NO_ALIASES",
    "DraftAliases",
    "RaggedTokenMerger",
    "RejectionSamplingStrategy",
    "SpecDecodeGraphInputs",
    "SpecDecodeInputTypeSpec",
    "SpecDecodeTailValues",
    "SpeculativeConfig",
    "SpeculativeMethod",
    "VerifyWidthRange",
    "build_spec_decode_input_types",
    "decode_spec_decode_input_values",
    "decode_spec_decode_tail",
    "ragged_token_merger",
    "validate_draft_state_dict",
]
