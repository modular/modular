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

A pure function of :class:`~.model_config.DeepseekV41Config`: no device, no
weights, no graph. Transcribed from vLLM ``72d59adc2c76de8f77aad640c7a04d663
a970735``'s ``DeepseekV4Attention.__init__`` (``vllm/models/deepseek_v41/
attention.py``), ``deepseek_v41_layer_type`` (``sparse_mla.py``), and
``build_deepseek_v4_rope`` (``common/rope.py``).

CSA2 cross-layer reuse, the indexer formula and candidate pool, the KV cache
leaves, the KV cache memory planner, the weight adapter, and norms/RoPE all
consume this resolver's output; call it once at config-resolution time and
thread the result through as plain data, rather than recomputing these rules
ad hoc at each call site.

This module also owns :func:`block_group`, the shared-subgraph body a layer
runs. The resolver yields one row per trunk layer plus one per DSpark draft
stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model_config import DeepseekV41Config

# vLLM's own layer-type vocabulary (its FlashMLA tile-scheduler plan name),
# keyed only on compress_ratio -- distinct from `attention_mode` below, which
# additionally carries source/consumer role. `sparse_mla.py:36-50`.
_VLLM_LAYER_TYPES = {0: "swaonly", 1: "c1a", 2: "c2a"}


@dataclass
class LayerInfo:
    """Describes one trunk layer or DSpark stage.

    A trunk layer has ``layer_id`` 0..``num_hidden_layers - 1``. A DSpark
    stage has ``layer_id >= num_hidden_layers`` and ``dspark_stage`` 0/1/2.
    """

    layer_id: int
    dspark_stage: int | None
    compress_ratio: int
    vllm_layer_type: str
    attention_mode: str
    kv_source_layer_id: int | None
    index_source_layer_id: int | None
    candidate_role: str
    side: str
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
    """The plan's vocabulary layered on vLLM's booleans.

    These four branches are exhaustive over ``(is_kv_source,
    is_index_source)``: the last, ``not is_index_source``, already covers
    both ``is_kv_source`` values whenever ``is_index_source`` is ``False``,
    so "a KV source that is not an index source" falls out as ``Reuse``
    rather than a distinct case -- matching the oracle's own
    ``attention_mode``, which has the same structure and the same dead final
    branch.
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
    """Transcribed line-for-line from ``attention.py:322-351,460-461``.

    Not a standalone vLLM function -- it lives inline in
    ``DeepseekV4Attention.__init__``, which also builds quantized linears, a
    paged KV cache, and a compiled-graph registration, none of which
    construct without a full ``VllmConfig``, process group, and weights.
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
        # Non-strict ``<=``: a layer at or below itself is its own source
        # (layer 2's KV source is layer 2), and the candidate-source layer
        # 20 owns the KV that the Reindex layers 24-36 rescore.
        kv_source_layer_id = max(s for s in kv_source_layers if s <= layer_id)
        index_source_layer_id = max(
            s for s in index_source_layers if s <= layer_id
        )

    # Gated on `is_index_source`: the indexer -- and so any candidate role --
    # is only constructed `if self.is_index_source`.
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
    """Transcribed from ``build_deepseek_v4_rope`` (``rope.py:9-53``).

    Sliding-window-only layers (``compress_ratio == 0``) use plain
    ``rope_theta`` with factor 1.0 -- no YaRN; compressed layers switch to
    ``compress_rope_theta`` with the YaRN factor from ``rope_scaling`` (16
    for the published checkpoint). A compressed layer whose
    ``rope_scaling`` sets ``rope_type: default`` collapses to factor 1.0
    (plain) while keeping ``compress_rope_theta``; the real checkpoint's
    ``rope_scaling`` has no ``attention_factor`` key and always takes the
    YaRN leg.
    """
    rope_scaling = config.rope_scaling or {}
    rope_type = rope_scaling.get("rope_type", "default")

    if compress_ratio > 0:
        theta = config.compress_rope_theta
        factor = (
            rope_scaling.get("factor", 1.0) if rope_type != "default" else 1.0
        )
    else:
        # vLLM still resolves this leg to `deepseek_yarn`, not a distinct
        # "plain" rope_type string, but force-overrides factor=1.0, which is
        # functionally no scaling.
        theta = config.rope_theta
        factor = 1.0

    regime = "plain" if factor == 1.0 else "yarn"
    return regime, theta, factor


def _validate_layer_mapping(
    config: DeepseekV41Config, include_drafter: bool
) -> None:
    """TokenSpeed's ``v41_layer_mapping`` validity rules, as early asserts.

    Ported in full from ``tokenspeed``'s
    ``layers/attention/deepseek_v41_geometry.py:158-200``: ratios at
    least one per trunk layer; owners/sources
    sorted, unique and in range; owners a subset of index sources
    with the candidate source among them; ratios in {0, 1, 2}; every
    compressed layer resolving an owner and index source at-or-below itself
    with a matching ratio; SWA-only layers never owning; and every Reindex
    layer sharing the candidate source's global address domain. Synthetic
    test configs must satisfy the same contract the production mapping
    does -- combinations the production rules disallow (an owner on an
    SWA-only layer, a KV owner that is not an index source) are exercised
    through :func:`_attention_mode` directly instead.

    The rules cover the trunk rows only. With ``include_drafter`` the
    trailing DSpark rows must each be window-only (ratio 0), the one shape a
    stage resolves to; without it they are never read, so a plain trunk
    boot does not depend on rows only the drafter uses.
    """
    ratios = tuple(config.compress_ratios)
    n = len(ratios)
    trunk = config.num_hidden_layers
    if n < trunk:
        raise ValueError(
            f"V4.1 compress_ratios must hold one entry per trunk layer "
            f"({config.num_hidden_layers}), got {n}"
        )
    owners = tuple(config.kv_source_layer_ids)
    sources = tuple(config.index_source_layer_ids)
    candidate_source = config.candidate_source_layer_id
    for name, layers in (("owners", owners), ("sources", sources)):
        if tuple(sorted(set(layers))) != layers or any(
            i < 0 or i >= trunk for i in layers
        ):
            raise ValueError(
                f"V4.1 {name} must be sorted unique trunk layer IDs"
            )
    if not set(owners).issubset(sources) or (
        candidate_source != -1 and candidate_source not in owners
    ):
        raise ValueError(
            "V4.1 owners must index; candidate source must own global KV"
        )
    for layer, ratio in enumerate(ratios[:trunk]):
        if ratio not in (0, 1, 2):
            raise ValueError("V4.1 compression ratios must be 0, 1, or 2")
        owner = (
            max((i for i in owners if i <= layer), default=-1) if ratio else -1
        )
        source = (
            max((i for i in sources if i <= layer), default=-1) if ratio else -1
        )
        if ratio and (
            owner < 0
            or source < owner
            or ratios[owner] != ratio
            or ratios[source] != ratio
        ):
            raise ValueError(
                f"V4.1 layer {layer} has incompatible owner/index source"
            )
        if layer in owners + sources and not ratio:
            raise ValueError(
                "SWA-only layers cannot own global KV or index selections"
            )
        if (
            source != owner
            and owner >= 0
            and (owner != candidate_source or source < candidate_source)
        ):
            raise ValueError(
                "Reindex must use the candidate source's KV address domain"
            )
    if include_drafter:
        for stage, ratio in enumerate(ratios[trunk:]):
            if ratio != 0:
                raise ValueError(
                    f"V4.1 DSpark stage {stage} must be window-only "
                    f"(compress ratio 0), got {ratio}"
                )


def resolve(
    config: DeepseekV41Config, include_drafter: bool = True
) -> list[LayerInfo]:
    """Resolves one :class:`LayerInfo` per trunk layer plus DSpark stage.

    ``config.compress_ratios`` has one entry per layer id (trunk layers
    followed by DSpark stages); this iterates it directly rather than
    taking a separate stage count, so it stays correct if DSpark ever
    changes its stage count -- but at least ``num_hidden_layers``
    entries are required, so a truncated list is a loud error rather
    than a partial trunk. DSpark stages resolve with
    ``side="drafter"`` and ratio 0 (SWA-only window attention).

    ``include_drafter=False`` resolves the trunk rows alone, for a model
    that never builds the drafter.
    """
    _validate_layer_mapping(config, include_drafter)

    # The plan's own convention, not a vLLM config field: the encoder/decoder
    # split coincides with the last KV-source layer for this checkpoint. Kept
    # derived (not a hardcoded layer id) so it re-resolves if
    # `kv_source_layer_ids` ever changes.
    # A config with no KV sources (every layer window-only) has no
    # encoder/decoder split; every layer sits past the (empty) start.
    decoder_start = max(config.kv_source_layer_ids, default=0)
    engram_layer_ids = set(config.engram_layer_ids)

    rows: list[LayerInfo] = []
    ratios = list(config.compress_ratios)
    if not include_drafter:
        ratios = ratios[: config.num_hidden_layers]
    for layer_id, compress_ratio in enumerate(ratios):
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
                vllm_layer_type=_VLLM_LAYER_TYPES[compress_ratio],
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
