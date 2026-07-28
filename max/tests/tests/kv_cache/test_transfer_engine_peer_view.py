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

"""Unit tests for KVTransferEngine's peer-view shape resolver.

``resolve_peer_view`` decides whether a local or remote ``[dp][tp]`` grid
is reinterpreted as ``[dp*tp][1]`` so a prefill worker at (DP=m, TP=n)
can connect to a decode worker at (DP=m*n, TP=1).
"""

from __future__ import annotations

import pytest
from max.pipelines.kv_cache.paged_kv_cache.transfer_engine import (
    connect_pairing,
    resolve_peer_view,
    transfer_shard_pairing,
)


@pytest.mark.parametrize(
    "name,local_dp,local_tp,local_rep,remote_dp,remote_tp,remote_rep,"
    "flatten_local,flatten_remote,effective_dp",
    [
        ("homogeneous_match", 2, 4, True, 2, 4, True, False, False, 2),
        ("local_flattens_mla_to_dp", 1, 8, True, 8, 1, True, True, False, 8),
        ("remote_flattens_dp_to_mla", 8, 1, True, 1, 8, True, False, True, 8),
        ("homogeneous_non_mla", 4, 2, False, 4, 2, False, False, False, 4),
    ],
)
def test_resolve_peer_view_accepted(
    name: str,
    local_dp: int,
    local_tp: int,
    local_rep: bool,
    remote_dp: int,
    remote_tp: int,
    remote_rep: bool,
    flatten_local: bool,
    flatten_remote: bool,
    effective_dp: int,
) -> None:
    view = resolve_peer_view(
        local_dp=local_dp,
        local_tp=local_tp,
        local_replicate=local_rep,
        remote_dp=remote_dp,
        remote_tp=remote_tp,
        remote_replicate=remote_rep,
    )
    assert view.flatten_local is flatten_local
    assert view.flatten_remote is flatten_remote
    assert view.effective_dp == effective_dp


@pytest.mark.parametrize(
    "name,local_dp,local_tp,local_rep,remote_dp,remote_tp,remote_rep",
    [
        # Heterogeneous shapes without MLA replication on either side.
        ("neither_replicates", 1, 8, False, 8, 1, False),
        # Local dp*tp=4 but remote dp=8.
        ("dp_product_mismatch", 1, 4, True, 8, 1, True),
        # Both sides TP>1 with mismatched shapes.
        ("both_tp_gt_1_mismatch", 2, 4, True, 4, 2, True),
        # TP=1 sides claiming to replicate (nonsensical; caller bug).
        ("tp1_replicate_nonsense", 8, 1, True, 1, 1, True),
    ],
)
def test_resolve_peer_view_rejected(
    name: str,
    local_dp: int,
    local_tp: int,
    local_rep: bool,
    remote_dp: int,
    remote_tp: int,
    remote_rep: bool,
) -> None:
    with pytest.raises(ValueError, match="Incompatible"):
        resolve_peer_view(
            local_dp=local_dp,
            local_tp=local_tp,
            local_replicate=local_rep,
            remote_dp=remote_dp,
            remote_tp=remote_tp,
            remote_replicate=remote_rep,
        )


# ---------------------------------------------------------------------------
# connect_pairing: physical (local_replica, local_shard, remote_replica,
# remote_shard) index quads for connect/disconnect/cleanup.
# ---------------------------------------------------------------------------


def _pairing(
    local_dp: int,
    local_tp: int,
    local_rep: bool,
    remote_dp: int,
    remote_tp: int,
    remote_rep: bool,
) -> list[tuple[int, int, int, int]]:
    view = resolve_peer_view(
        local_dp=local_dp,
        local_tp=local_tp,
        local_replicate=local_rep,
        remote_dp=remote_dp,
        remote_tp=remote_tp,
        remote_replicate=remote_rep,
    )
    return connect_pairing(view, local_dp, local_tp, remote_dp, remote_tp)


def test_connect_pairing_homogeneous_dp2() -> None:
    """dp=2,tp=2: full cartesian product of replicas, shard-wise zip."""
    assert _pairing(2, 2, False, 2, 2, False) == [
        (0, 0, 0, 0),
        (0, 1, 0, 1),
        (0, 0, 1, 0),
        (0, 1, 1, 1),
        (1, 0, 0, 0),
        (1, 1, 0, 1),
        (1, 0, 1, 0),
        (1, 1, 1, 1),
    ]


def test_connect_pairing_flatten_local() -> None:
    """Local (dp=1,tp=2,MLA) flattens to [dp*tp][1] against remote (dp=2,tp=1)."""
    assert _pairing(1, 2, True, 2, 1, True) == [
        (0, 0, 0, 0),
        (0, 0, 1, 0),
        (0, 1, 0, 0),
        (0, 1, 1, 0),
    ]


def test_connect_pairing_flatten_remote() -> None:
    """Remote (dp=1,tp=2,MLA) flattens against local (dp=2,tp=1)."""
    assert _pairing(2, 1, True, 1, 2, True) == [
        (0, 0, 0, 0),
        (0, 0, 0, 1),
        (1, 0, 0, 0),
        (1, 0, 0, 1),
    ]


def test_connect_pairing_dp2tp4_to_dp8_mla() -> None:
    """DP2/TP4 MLA prefill -> DP8/TP1 decode: 8x8 effective mesh."""
    pairs = _pairing(2, 4, True, 8, 1, True)
    assert len(pairs) == 64
    # Local effective replica e maps to physical (e // 4, e % 4); it pairs
    # with every one of the 8 remote replicas, remote shard always 0.
    assert pairs[:8] == [(0, 0, r, 0) for r in range(8)]
    assert pairs[8:16] == [(0, 1, r, 0) for r in range(8)]
    # Last local effective replica is physical (1, 3).
    assert pairs[-8:] == [(1, 3, r, 0) for r in range(8)]


# ---------------------------------------------------------------------------
# transfer_shard_pairing: (source_shard, dest_shard) pairs for one transfer.
# ---------------------------------------------------------------------------


def test_transfer_shard_pairing_homogeneous() -> None:
    """No flatten, equal TP: 1:1 shard pairing."""
    pairs = transfer_shard_pairing(flatten_source=False, source_tp=2, dest_tp=2)
    assert pairs == [(0, 0), (1, 1)]


def test_transfer_shard_pairing_flatten_source() -> None:
    """Flatten source collapses to shard 0 (dest TP=1)."""
    pairs = transfer_shard_pairing(flatten_source=True, source_tp=4, dest_tp=1)
    assert pairs == [(0, 0)]


def test_transfer_shard_pairing_fanout() -> None:
    """Single source shard fans out to every destination shard."""
    pairs = transfer_shard_pairing(flatten_source=False, source_tp=1, dest_tp=4)
    assert pairs == [(0, 0), (0, 1), (0, 2), (0, 3)]
    # local_shards_used semantics: source side for a send.
    assert [src for src, _ in pairs] == [0, 0, 0, 0]
