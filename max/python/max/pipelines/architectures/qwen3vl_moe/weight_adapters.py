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
"""Weight rules for Qwen3-VL and Qwen3-VL-MoE."""

from __future__ import annotations

from collections.abc import Sequence

from max.pipelines.lib import WeightRenaming, WeightRule
from transformers import PretrainedConfig

QWEN3VL_WEIGHT_RULES: tuple[WeightRule, ...] = (
    WeightRenaming(r"^model\.language_model\.", ""),
    WeightRenaming(r"^model\.visual\.", "vision_encoder."),
    WeightRenaming(
        r"^vision_encoder\.pos_embed\.", "vision_encoder.pos_embed.embedding."
    ),
    WeightRenaming(
        r"^(vision_encoder\.blocks\.\d+\.attn\.)qkv\.", r"\1qkv_proj."
    ),
    WeightRenaming(r"\.mlp\.gate\.weight$", ".mlp.gate.gate_score.weight"),
)
"""Maps a Qwen3-VL checkpoint onto its decoder and vision tower.

The modules declare every weight in the dtype and shape the checkpoint
stores it, so the rules only rename: the Conv3D patch projection is
flattened in the graph.
"""


def qwen3vl_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns :data:`QWEN3VL_WEIGHT_RULES`, which no config changes."""
    del huggingface_config
    return QWEN3VL_WEIGHT_RULES
