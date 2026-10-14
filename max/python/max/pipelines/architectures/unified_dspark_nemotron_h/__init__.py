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
"""Unified DSpark speculative decoding for Nemotron-H."""

from .arch import qwen3_dspark_draft_arch, unified_dspark_nemotron_h_arch
from .dspark_qwen3 import DSparkQwen3, DSparkQwen3DraftConfig
from .model import UnifiedDSparkNemotronHInputs, UnifiedDSparkNemotronHModel
from .model_config import UnifiedDSparkNemotronHConfig
from .unified_dspark_nemotron_h import (
    UnifiedDSparkNemotronH,
    UnifiedDSparkNemotronHModule,
)

__all__ = [
    "DSparkQwen3",
    "DSparkQwen3DraftConfig",
    "UnifiedDSparkNemotronH",
    "UnifiedDSparkNemotronHConfig",
    "UnifiedDSparkNemotronHInputs",
    "UnifiedDSparkNemotronHModel",
    "UnifiedDSparkNemotronHModule",
    "qwen3_dspark_draft_arch",
    "unified_dspark_nemotron_h_arch",
]
