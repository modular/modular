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
"""Per-layer attention-mode resolver for DeepSeek-V4.1-Flash.

Derives each layer's attention structure from the model config alone: its
compression ratio, which earlier layers it shares KV and indexer results
with, its RoPE settings and whether it has Engram or an indexer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model_config import DeepseekV41Config


@dataclass
class LayerInfo:
    """Attention structure of one trunk layer or DSpark draft stage.

    Trunk layers have ``layer_id < num_hidden_layers``; DSpark stages follow
    them, numbered by ``dspark_stage`` from 0.
    """

    layer_id: int
    dspark_stage: int | None
    compress_ratio: int
    """KV compression ratio; 0 means sliding-window only."""
    attention_mode: str
    """How this layer gets its KV and index results.

    ``"SWA-only"`` (sliding window only), ``"Full"`` (a KV and index source),
    ``"Reindex"`` (an index source only) or ``"Reuse"`` (not an index source).
    """
    kv_source_layer_id: int | None
    """Layer whose KV this layer reads; ``None`` if sliding-window only."""
    index_source_layer_id: int | None
    """Layer whose indexer results this layer reads.

    ``None`` for sliding-window-only layers.
    """
    candidate_role: str
    """Whether this layer produces or uses indexer candidate blocks.

    ``"source"`` if its indexer produces candidate blocks for later layers,
    ``"consumer"`` if it uses them, else ``"none"``.
    """
    side: str
    """``"encoder"`` or ``"decoder"`` for trunk layers, else ``"drafter"``."""
    has_engram: bool
    swa_only: bool
    rope_regime: str
    rope_theta: float
    rope_yarn_factor: float
    has_indexer: bool
    indexer_topk: int | None


def _attention_mode(
    compress_ratio: int, is_kv_source: bool, is_index_source: bool
) -> str:
    """Returns the named attention mode for a layer's source flags.

    A KV source that is not an index source resolves to ``Reuse``.
    """
    if compress_ratio == 0:
        return "SWA-only"
    if is_kv_source and is_index_source:
        return "Full"
    if is_index_source and not is_kv_source:
        return "Reindex"
    return "Reuse"


def _resolve_source_and_candidate(
    config: DeepseekV41Config, layer_id: int, compress_ratio: int
) -> tuple[bool, bool, int | None, int | None, str]:
    """Resolves a layer's source flags, source ids and candidate role.

    A compressed layer reads KV and indexer results from the nearest source
    layer at or before it. Only trunk layers can be sources.
    """
    kv_source_layers = tuple(config.kv_source_layer_ids)
    index_source_layers = tuple(config.index_source_layer_ids)
    candidate_source_layer = config.candidate_source_layer_id

    is_backbone = layer_id < config.num_hidden_layers
    is_kv_source = is_backbone and layer_id in kv_source_layers
    is_index_source = is_backbone and layer_id in index_source_layers

    kv_source_layer_id = None
    index_source_layer_id = None
    if compress_ratio > 0:
        kv_source_layer_id = max(s for s in kv_source_layers if s <= layer_id)
        index_source_layer_id = max(
            s for s in index_source_layers if s <= layer_id
        )

    # Only index sources build an indexer, so only they get a candidate role.
    is_candidate_source = is_index_source and layer_id == candidate_source_layer
    uses_candidates = is_index_source and 0 <= candidate_source_layer < layer_id
    if is_candidate_source:
        candidate_role = "source"
    elif uses_candidates:
        candidate_role = "consumer"
    else:
        candidate_role = "none"

    return (
        is_kv_source,
        is_index_source,
        kv_source_layer_id,
        index_source_layer_id,
        candidate_role,
    )


def _rope_regime(
    config: DeepseekV41Config, compress_ratio: int
) -> tuple[str, float, float]:
    """Returns a layer's RoPE ``(regime, theta, factor)``.

    Compressed layers use ``compress_rope_theta`` and the configured scaling
    factor; any factor other than 1.0 is treated as YaRN.
    """
    rope_scaling = config.rope_scaling or {}
    rope_type = rope_scaling.get("rope_type", "default")

    if compress_ratio > 0:
        theta = config.compress_rope_theta
        factor = (
            rope_scaling.get("factor", 1.0) if rope_type != "default" else 1.0
        )
    else:
        # Sliding-window-only layers never scale RoPE.
        theta = config.rope_theta
        factor = 1.0

    regime = "plain" if factor == 1.0 else "yarn"
    return regime, theta, factor


def resolve(config: DeepseekV41Config) -> list[LayerInfo]:
    """Resolves one :class:`LayerInfo` per trunk layer plus DSpark stage.

    Args:
        config: The model config. ``compress_ratios`` has one entry per
            layer id: trunk layers, then DSpark stages.

    Returns:
        One :class:`LayerInfo` per entry of ``config.compress_ratios``.
    """
    # Trunk layers from the last KV-source layer onward form the decoder.
    decoder_start = max(config.kv_source_layer_ids)
    engram_layer_ids = set(config.engram_layer_ids)

    rows: list[LayerInfo] = []
    for layer_id, compress_ratio in enumerate(config.compress_ratios):
        (
            is_kv_source,
            is_index_source,
            kv_source_layer_id,
            index_source_layer_id,
            candidate_role,
        ) = _resolve_source_and_candidate(config, layer_id, compress_ratio)
        rope_regime, rope_theta, rope_yarn_factor = _rope_regime(
            config, compress_ratio
        )

        if layer_id < config.num_hidden_layers:
            side = "encoder" if layer_id < decoder_start else "decoder"
            dspark_stage = None
        else:
            side = "drafter"
            dspark_stage = layer_id - config.num_hidden_layers

        rows.append(
            LayerInfo(
                layer_id=layer_id,
                dspark_stage=dspark_stage,
                compress_ratio=compress_ratio,
                attention_mode=_attention_mode(
                    compress_ratio, is_kv_source, is_index_source
                ),
                kv_source_layer_id=kv_source_layer_id,
                index_source_layer_id=index_source_layer_id,
                candidate_role=candidate_role,
                side=side,
                has_engram=layer_id in engram_layer_ids,
                swa_only=compress_ratio == 0,
                rope_regime=rope_regime,
                rope_theta=rope_theta,
                rope_yarn_factor=rope_yarn_factor,
                has_indexer=is_index_source,
                indexer_topk=config.index_topk if is_index_source else None,
            )
        )
    return rows
