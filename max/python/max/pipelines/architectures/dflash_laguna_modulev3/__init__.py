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
"""The DFlash drafter of a Laguna target, on ModuleV3."""

from .arch import dflash_laguna_modulev3_draft_arch
from .dflash_laguna import DFlashLaguna
from .model_config import (
    SERVING_BLOCK_SIZE,
    DFlashLagunaConfig,
    DFlashLagunaDraftConfig,
    dflash_laguna_draft_width,
)

__all__ = [
    "SERVING_BLOCK_SIZE",
    "DFlashLaguna",
    "DFlashLagunaConfig",
    "DFlashLagunaDraftConfig",
    "dflash_laguna_draft_width",
    "dflash_laguna_modulev3_draft_arch",
]
