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
# Ported to ModuleV3 from
# max/python/max/pipelines/architectures/dflash2_qwen3_5/__init__.py
# at main 4df8111afba.
"""The DFlash2 drafter of a Qwen3.5 target, on ModuleV3."""

from .arch import dflash2_qwen3_5_modulev3_draft_arch
from .dflash2_qwen3_5 import DFlash2Qwen3_5
from .model import (
    DFlash2Qwen3_5DraftArchConfig,
    dflash2_draft_registration,
)
from .model_config import (
    DRAFT_SLIDING_WINDOW,
    Dflash2DraftHFConfig,
    DFlash2Qwen3_5Config,
    dflash2_draft_width,
    parse_dflash2_draft_hf_config,
)
from .proposer import dflash2_block_proposer

__all__ = [
    "DRAFT_SLIDING_WINDOW",
    "DFlash2Qwen3_5",
    "DFlash2Qwen3_5Config",
    "DFlash2Qwen3_5DraftArchConfig",
    "Dflash2DraftHFConfig",
    "dflash2_block_proposer",
    "dflash2_draft_registration",
    "dflash2_draft_width",
    "dflash2_qwen3_5_modulev3_draft_arch",
    "parse_dflash2_draft_hf_config",
]
