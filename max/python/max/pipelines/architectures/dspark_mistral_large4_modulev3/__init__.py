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
"""Mistral Large 4's in-checkpoint DSpark drafter as a draft architecture.

The drafter is ``mtp_dspark.*`` in the target's own checkpoint, so this
architecture owns no pipeline model and no checkpoint of its own: the pairing
builds it beside the target, hands it the target's embedding and LM head, and
compiles the two as one fused graph named
``UnifiedDSparkMistralLarge4Module``.
"""

from .arch import dspark_mistral_large4_modulev3_draft_arch
from .model import (
    DSparkMistralLarge4DraftArchConfig,
    DSparkMistralLarge4DraftConfig,
)

__all__ = [
    "DSparkMistralLarge4DraftArchConfig",
    "DSparkMistralLarge4DraftConfig",
    "dspark_mistral_large4_modulev3_draft_arch",
]
