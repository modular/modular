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
"""Weight rules for Qwen3.5 models."""

from __future__ import annotations

from collections.abc import Sequence

from max.graph.weights import WeightsAdapter
from max.pipelines.lib import (
    CheckpointRole,
    Ignore,
    WeightEntry,
    WeightRenaming,
    WeightRule,
    WeightTransform,
)
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


def qwen3_5_weight_entry(adapter: WeightsAdapter | None) -> WeightEntry:
    """Returns the entry that names a Qwen3.5 checkpoint through ``adapter``.

    Args:
        adapter: The weight transform built from the architecture's rules.

    Raises:
        ValueError: If ``adapter`` is not a :class:`WeightTransform`.
    """
    if not isinstance(adapter, WeightTransform):
        raise ValueError(
            "Qwen3.5 loads its checkpoint through its architecture's weight"
            " rules, but the model has no WeightTransform"
        )
    return WeightEntry(CheckpointRole.TARGET, adapter)
