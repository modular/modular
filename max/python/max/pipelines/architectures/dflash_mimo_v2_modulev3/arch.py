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
"""Supported architecture entry for the ModuleV3 DFlash MiMo V2 drafter."""

from dataclasses import replace

from ..dflash_mimo_v2.arch import dflash_mimo_v2_draft_arch
from .model import DFlashMiMoV2ModuleV3DraftArchConfig

# The same checkpoint as the graph-API drafter, drafting for a ModuleV3
# MiMo-V2 target.
dflash_mimo_v2_modulev3_draft_arch = replace(
    dflash_mimo_v2_draft_arch,
    name="DFlashMiMoV2DraftModel_ModuleV3",
    config=DFlashMiMoV2ModuleV3DraftArchConfig,
)
