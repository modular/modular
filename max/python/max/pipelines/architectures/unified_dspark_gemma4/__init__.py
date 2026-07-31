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
"""Unified DSpark speculative decoding for Gemma4."""

from .arch import gemma4_dspark_draft_arch, unified_dspark_gemma4_arch
from .dspark_gemma4 import (
    DSparkGemma4,
    DSparkGemma4Attention,
    DSparkGemma4DecoderLayer,
    DSparkGemma4DraftConfig,
    DSparkMarkovHead,
)
from .model import UnifiedDSparkGemma4Inputs, UnifiedDSparkGemma4Model
from .model_config import UnifiedDSparkGemma4Config
from .unified_dspark_gemma4 import UnifiedDSparkGemma4

__all__ = [
    "DSparkGemma4",
    "DSparkGemma4Attention",
    "DSparkGemma4DecoderLayer",
    "DSparkGemma4DraftConfig",
    "DSparkMarkovHead",
    "UnifiedDSparkGemma4",
    "UnifiedDSparkGemma4Config",
    "UnifiedDSparkGemma4Inputs",
    "UnifiedDSparkGemma4Model",
    "gemma4_dspark_draft_arch",
    "unified_dspark_gemma4_arch",
]
