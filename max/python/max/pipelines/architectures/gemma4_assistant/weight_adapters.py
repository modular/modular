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
"""Maps Gemma4Assistant checkpoint names onto the MAX tree."""

from __future__ import annotations

from max.pipelines.lib import WeightRenaming, WeightRule

GEMMA4_ASSISTANT_WEIGHT_RULES: tuple[WeightRule, ...] = (
    # Its own embedding, which only its tied head reads: ``embed_tokens`` is
    # the target's.
    WeightRenaming(r"^model\.embed_tokens\.", "draft_embed_tokens."),
    WeightRenaming(r"^model\.", ""),
)
"""Names the assistant checkpoint's weights as its module does."""
