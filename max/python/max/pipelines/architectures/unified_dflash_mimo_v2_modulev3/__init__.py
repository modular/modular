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
"""MiMo-V2.6-Flash with its DFlash drafter on ModuleV3: the fused speculative
graph."""

from .arch import (
    unified_dflash_mimo_v2_modulev3_arch,
    unified_dflash_mimo_v2_modulev3_speculator,
)
from .drafter import DrafterExport
from .model import UnifiedDflashMiMoV2ModuleV3Model
from .unified_dflash_mimo_v2 import (
    GRAPH_NAME,
    UnifiedDflashMiMoV2ModuleV3,
    UnifiedDflashMiMoV2Spec,
)

__all__ = [
    "GRAPH_NAME",
    "DrafterExport",
    "UnifiedDflashMiMoV2ModuleV3",
    "UnifiedDflashMiMoV2ModuleV3Model",
    "UnifiedDflashMiMoV2Spec",
    "unified_dflash_mimo_v2_modulev3_arch",
    "unified_dflash_mimo_v2_modulev3_speculator",
]
