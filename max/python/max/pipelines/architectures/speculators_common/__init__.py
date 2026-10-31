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
"""Shared format helpers for vLLM speculators-format DSpark checkpoints.

The format-level pieces of the speculators DSpark contract — config-schema
parsing, the d2t vocabulary mapping, and checkpoint weight handling —
reused by every speculators-format DSpark draft. The generic
``DSparkDraftModel`` registration and the draft nn.Module live in the
``dspark_draft`` package.
"""

from .d2t import map_draft_to_target_vocab
from .draft_config import (
    DSparkSpeculatorsDraftArchConfig,
    DSparkSpeculatorsDraftConfig,
    construct_draft_kv_params,
)
from .weight_adapters import (
    SKIPPED_DRAFT_KEYS,
    SKIPPED_DRAFT_PREFIXES,
    validate_draft_checkpoint_weights,
)

__all__ = [
    "SKIPPED_DRAFT_KEYS",
    "SKIPPED_DRAFT_PREFIXES",
    "DSparkSpeculatorsDraftArchConfig",
    "DSparkSpeculatorsDraftConfig",
    "construct_draft_kv_params",
    "map_draft_to_target_vocab",
    "validate_draft_checkpoint_weights",
]
