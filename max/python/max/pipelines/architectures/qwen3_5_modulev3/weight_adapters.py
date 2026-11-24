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
# Copied from max/python/max/pipelines/architectures/qwen3_5/weight_adapters.py
# at main 4df8111afba.
"""Weight rules for Qwen3.5 models."""

from __future__ import annotations

from collections.abc import Sequence

from max.pipelines.lib import Ignore, WeightRenaming, WeightRule
from transformers import PretrainedConfig

QWEN3_5_WEIGHT_RULES: tuple[WeightRule, ...] = (
    Ignore(
        r"^mtp\.",
        reason="the multi-token prediction head, which only the"
        " speculative-decoding pipelines load",
    ),
    WeightRenaming(r"^model\.visual\.", "vision_encoder."),
    WeightRenaming(
        r"^vision_encoder\.pos_embed\.", "vision_encoder.pos_embed.embedding."
    ),
    WeightRenaming(
        r"^(vision_encoder\.blocks\.\d+\.attn\.)qkv\.", r"\1qkv_proj."
    ),
    # Multimodal checkpoints nest the language model, text-only ones do not.
    WeightRenaming(r"^model\.language_model\.", ""),
    WeightRenaming(r"^model\.", ""),
)
"""Maps a Qwen3.5 checkpoint onto its language and vision modules.

The graphs declare every weight in the dtype and shape the checkpoint stores
it, so the rules only rename: the gated-DeltaNet scalars and the Conv3D patch
projection are cast and reshaped in the graph.
"""

QWEN3_5_MTP_WEIGHT_RULES: tuple[WeightRule, ...] = (
    Ignore(
        r"^(?:model|lm_head)\.",
        reason="the target model's weights, which the target's rules name",
    ),
    WeightRenaming(r"^mtp\.", ""),
)
"""Names the multi-token prediction head's weights as its module does.

The head ships in the target's checkpoint, which speculative decoding also
loads as the draft's.
"""


def qwen3_5_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns :data:`QWEN3_5_WEIGHT_RULES`, which no config changes."""
    del huggingface_config
    return QWEN3_5_WEIGHT_RULES
