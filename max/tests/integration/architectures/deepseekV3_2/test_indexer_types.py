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
"""Unit tests for the DeepSeek Sparse Attention indexer schedule resolution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from max.pipelines.architectures.deepseekV3_2.deepseekV3_2 import (
    _validate_indexer_types,
)
from max.pipelines.architectures.deepseekV3_2.model_config import (
    indexer_cache_layer_map,
    num_indexer_cache_layers,
    resolve_indexer_types,
)


def test_explicit_indexer_types_passthrough() -> None:
    """An explicit ``indexer_types`` list is returned verbatim."""
    cfg = SimpleNamespace(indexer_types=["full", "shared", "shared", "full"])
    assert resolve_indexer_types(cfg, 4) == [
        "full",
        "shared",
        "shared",
        "full",
    ]


def test_missing_fields_defaults_to_all_full() -> None:
    """Models without any indexer-share fields (DeepSeek V3.2, GLM-5.1)."""
    cfg = SimpleNamespace()
    assert resolve_indexer_types(cfg, 5) == ["full"] * 5


def test_freq_offset_schedule_matches_glm_5_2() -> None:
    """``index_topk_freq`` / ``index_skip_topk_offset`` reproduce GLM-5.2.

    GLM-5.2 ships ``index_topk_freq=4`` and ``index_skip_topk_offset=3``; the
    derived full layers are ``0, 1, 2`` (dense) then every 4th from layer 6.
    """
    cfg = SimpleNamespace(
        indexer_types=None,
        index_topk_pattern=None,
        index_topk_freq=4,
        index_skip_topk_offset=3,
    )
    resolved = resolve_indexer_types(cfg, 16)
    full = [i for i, t in enumerate(resolved) if t == "full"]
    assert full == [0, 1, 2, 6, 10, 14]
    assert set(resolved) == {"full", "shared"}


def test_string_pattern_schedule() -> None:
    """A string ``index_topk_pattern`` maps ``F``/``S`` to full/shared."""
    cfg = SimpleNamespace(indexer_types=None, index_topk_pattern="FSSF")
    assert resolve_indexer_types(cfg, 4) == [
        "full",
        "shared",
        "shared",
        "full",
    ]


def test_validate_indexer_types_rejects_leading_shared() -> None:
    """A ``shared`` first layer has no preceding full layer to reuse."""
    cfg = SimpleNamespace(indexer_types=["shared", "full", "shared"])
    with pytest.raises(ValueError, match="indexer_types\\[0\\] must be 'full'"):
        _validate_indexer_types(cfg)  # type: ignore[arg-type]


def test_validate_indexer_types_accepts_leading_full_and_empty() -> None:
    """A full-first schedule, and the empty (all-full) schedule, are valid."""
    _validate_indexer_types(
        SimpleNamespace(indexer_types=["full", "shared", "shared"])  # type: ignore[arg-type]
    )
    _validate_indexer_types(SimpleNamespace(indexer_types=[]))  # type: ignore[arg-type]


def _glm_5_3_schedule() -> list[str]:
    cfg = SimpleNamespace(
        indexer_types=None,
        index_topk_pattern=None,
        index_topk_freq=4,
        index_skip_topk_offset=3,
    )
    return resolve_indexer_types(cfg, 78)


def test_indexer_cache_holds_only_full_layers() -> None:
    """GLM-5.3 runs the indexer in 21 of its 78 layers, so the indexer K
    cache needs 21 layers, not 78."""
    schedule = _glm_5_3_schedule()
    assert num_indexer_cache_layers(schedule, 78) == 21


def test_indexer_cache_slots_are_dense_and_ordered() -> None:
    """Each full layer gets the next slot; slots cover the cache exactly."""
    schedule = _glm_5_3_schedule()
    slots = indexer_cache_layer_map(schedule, 78)
    full_slots = [slots[i] for i, t in enumerate(schedule) if t == "full"]
    assert full_slots == list(range(21))
    assert all(slots[i] == 0 for i, t in enumerate(schedule) if t == "shared")


def test_empty_schedule_keeps_one_slot_per_layer() -> None:
    """An empty schedule (the MTP draft) means every layer is full."""
    assert num_indexer_cache_layers([], 3) == 3
    assert indexer_cache_layer_map([], 3) == [0, 1, 2]


def test_schedule_longer_than_layers_is_truncated() -> None:
    """A draft that shares the target's schedule only counts its own
    layers."""
    schedule = ["full", "shared", "full"]
    assert num_indexer_cache_layers(schedule, 1) == 1
    assert indexer_cache_layer_map(schedule, 1) == [0]
