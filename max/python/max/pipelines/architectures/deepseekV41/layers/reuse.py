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
"""Binds layers to KV caches and top-k selections owned by other layers.

A layer may read the caches written by an earlier source layer, or reuse the
top-k selection a source layer computed in the same forward pass. Caches are
looked up by the source layer's id. The selection is passed from layer to
layer and must never be recovered from persistent cache state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

from ..layer_resolver import LayerInfo

_CacheLeaf = TypeVar("_CacheLeaf")
_SelectionValue = TypeVar("_SelectionValue")
_CandidatePool = TypeVar("_CandidatePool")


@dataclass(frozen=True)
class CacheSourceBinding(Generic[_CacheLeaf]):
    """Persistent caches a compressed layer reads, and the layer owning them.

    Both caches belong to the KV-source layer: only a layer that compresses
    its own KV also produces index keys.
    """

    kv_source_layer_id: int
    main_kv_leaf: _CacheLeaf
    index_k_leaf: _CacheLeaf


def bind_cache_sources(
    layer: LayerInfo,
    main_kv_by_layer: Mapping[int, _CacheLeaf],
    index_k_by_layer: Mapping[int, _CacheLeaf],
) -> CacheSourceBinding[_CacheLeaf] | None:
    """Binds a layer to the main KV and index-key caches of its KV source.

    The index keys are looked up by ``kv_source_layer_id``, not
    ``index_source_layer_id``: an index-source layer that is not also a KV
    source scores its own queries against its KV source's index keys.

    Args:
        layer: The reading layer's resolved attention metadata.
        main_kv_by_layer: Main KV caches keyed by owning layer id.
        index_k_by_layer: Indexer key caches keyed by owning layer id.

    Returns:
        The binding, or ``None`` for an SWA-only layer.

    Raises:
        ValueError: If the layer's source ids are incomplete or its source
            caches are missing (an inconsistent graph or pipeline-parallel
            partition).
    """
    if layer.attention_mode == "SWA-only":
        return None
    kv_source = layer.kv_source_layer_id
    index_source = layer.index_source_layer_id
    if kv_source is None or index_source is None:
        raise ValueError(
            f"compressed layer {layer.layer_id} has incomplete source ids"
        )
    try:
        main_kv_leaf = main_kv_by_layer[kv_source]
        index_k_leaf = index_k_by_layer[kv_source]
    except KeyError as exc:
        raise ValueError(
            f"layer {layer.layer_id} cannot bind KV/index-K source layer "
            f"{kv_source}"
        ) from exc
    return CacheSourceBinding(
        kv_source_layer_id=kv_source,
        main_kv_leaf=main_kv_leaf,
        index_k_leaf=index_k_leaf,
    )


@dataclass(frozen=True)
class ForwardSelection(Generic[_SelectionValue, _CandidatePool]):
    """One layer's top-k selection for this forward, with its candidate pool."""

    source_layer_id: int
    indices: _SelectionValue
    candidate_source_layer_id: int | None = None
    candidate_pool: _CandidatePool | None = None


def selection_for_layer(
    layer: LayerInfo,
    current: ForwardSelection[_SelectionValue, _CandidatePool] | None,
    *,
    produced: _SelectionValue | None = None,
    candidate_pool: _CandidatePool | None = None,
) -> ForwardSelection[_SelectionValue, _CandidatePool] | None:
    """Advances this forward pass's top-k selection across one layer.

    Full/Reindex layers publish ``produced`` tagged with their own layer id;
    Reuse layers must receive the selection their ``index_source_layer_id``
    produced in this same forward. Candidate pools are opaque and carried by
    identity.

    Args:
        layer: The resolved layer being advanced across.
        current: The selection published by earlier layers, if any.
        produced: This layer's own selection (Full/Reindex only).
        candidate_pool: This layer's candidate pool (candidate sources only).

    Returns:
        The selection visible after this layer, or ``None`` for SWA-only.

    Raises:
        ValueError: If the layer publishes state its mode forbids, omits
            state it must publish, or receives a mismatched selection.
    """
    if layer.attention_mode == "SWA-only":
        if produced is not None or candidate_pool is not None:
            raise ValueError("SWA-only layers cannot publish CSA2 state")
        return None

    if layer.attention_mode in ("Full", "Reindex"):
        if produced is None:
            raise ValueError(
                f"{layer.attention_mode} layer {layer.layer_id} must publish "
                "the current-forward selection"
            )
        if layer.index_source_layer_id != layer.layer_id:
            raise ValueError(
                f"index-source layer {layer.layer_id} resolves to "
                f"{layer.index_source_layer_id}"
            )
        pool_source = (
            layer.layer_id
            if layer.candidate_role == "source"
            else current.candidate_source_layer_id
            if current is not None
            else None
        )
        pool = candidate_pool
        if layer.candidate_role == "source" and candidate_pool is None:
            raise ValueError(
                f"candidate-source layer {layer.layer_id} must publish its "
                "candidate pool"
            )
        if layer.candidate_role != "source" and current is not None:
            pool = current.candidate_pool
            pool_source = current.candidate_source_layer_id
        return ForwardSelection(
            source_layer_id=layer.layer_id,
            indices=produced,
            candidate_source_layer_id=pool_source,
            candidate_pool=pool,
        )

    if layer.attention_mode != "Reuse":
        raise ValueError(f"unsupported attention mode {layer.attention_mode!r}")
    if produced is not None or candidate_pool is not None:
        raise ValueError("Reuse layers cannot publish CSA2 state")
    if (
        current is None
        or current.source_layer_id != layer.index_source_layer_id
    ):
        actual = None if current is None else current.source_layer_id
        raise ValueError(
            f"Reuse layer {layer.layer_id} needs current selection from "
            f"layer {layer.index_source_layer_id}, got {actual}"
        )
    return current
