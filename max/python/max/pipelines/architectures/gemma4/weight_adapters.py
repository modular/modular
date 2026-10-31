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
"""Maps Gemma 4 checkpoint names onto the MAX tree."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from max.dtype import DType
from max.graph.weights import WeightData, Weights, WeightsAdapter
from max.pipelines.lib import (
    Bitcast,
    CheckpointRole,
    Ignore,
    WeightEntry,
    WeightRenaming,
    WeightRule,
    WeightTransform,
)
from transformers import PretrainedConfig

GEMMA4_VISION_PREFIXES = ("model.vision_tower.", "model.embed_vision.")

GEMMA4_WEIGHT_RULES: tuple[WeightRule, ...] = (
    Ignore(
        r"^model\.(embed_audio|vision_embedder)\.",
        reason="the unified checkpoints' audio and patch embedders, which"
        " MAX does not implement",
    ),
    Ignore(
        r"\.(k|v)_scale$",
        reason="modelopt FP8 KV-cache scales, which the BF16 KV cache does"
        " not read",
    ),
    WeightRenaming(r"^model\.language_model\.", ""),
    WeightRenaming(r"^language_model\.model\.", ""),
    WeightRenaming(
        r"\.router\.proj\.weight$", ".moe_block.gate.gate_score.weight"
    ),
    WeightRenaming(r"\.router\.scale$", ".moe_block.gate.scale"),
    WeightRenaming(
        r"\.router\.per_expert_scale$", ".moe_block.gate.per_expert_scale"
    ),
    WeightRenaming(
        r"\.pre_feedforward_layernorm_2\.weight$",
        ".moe_block.pre_expert_norm.weight",
    ),
    WeightRenaming(r"\.experts\.", ".moe_block.experts."),
    WeightRenaming(r"^model\.vision_tower\.", ""),
    WeightRenaming(r"^model\.embed_vision\.", "embed_vision."),
    WeightRenaming(r"\.linear\.", "."),
    Bitcast(r"\.weight_scale$", src=DType.uint8, dst=DType.float8_e8m0fnu),
)
"""Maps a Gemma 4 checkpoint onto its language and vision modules."""


def gemma4_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns :data:`GEMMA4_WEIGHT_RULES`, which no config changes."""
    del huggingface_config
    return GEMMA4_WEIGHT_RULES


def gemma4_weight_entry(adapter: WeightsAdapter | None) -> WeightEntry:
    """Returns the entry that names a Gemma 4 checkpoint through ``adapter``.

    Args:
        adapter: The weight transform built from the architecture's rules.

    Raises:
        ValueError: If ``adapter`` is not a :class:`WeightTransform`.
    """
    if not isinstance(adapter, WeightTransform):
        raise ValueError(
            "Gemma 4 loads its checkpoint through its architecture's weight"
            " rules, but the model has no WeightTransform"
        )
    return WeightEntry(CheckpointRole.TARGET, adapter)


def adapt_gemma4_checkpoint(
    adapter: WeightsAdapter | None, weights: Mapping[str, Weights]
) -> tuple[dict[str, WeightData], dict[str, WeightData]]:
    """Adapts a Gemma 4 checkpoint's language and vision weights apart.

    The graph API's towers load strictly, so each takes only its own
    weights. Rules act on each weight alone, so adapting each tower's part
    of the checkpoint gives that tower's part of the adapted one.

    Args:
        adapter: The weight transform built from the architecture's rules.
        weights: The checkpoint's weights, keyed by checkpoint name.

    Returns:
        The adapted language weights and the adapted vision weights.

    Raises:
        ValueError: If ``adapter`` is not a :class:`WeightTransform`.
    """
    transform = gemma4_weight_entry(adapter).transform
    language: dict[str, Weights] = {}
    vision: dict[str, Weights] = {}
    for name, weight in weights.items():
        tower = vision if name.startswith(GEMMA4_VISION_PREFIXES) else language
        tower[name] = weight
    return transform(language), transform(vision)
