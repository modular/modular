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

"""Tests for the read-only prefix-cache hit-count query.

Covers the two building blocks added for prefix-aware data-parallel routing:

- ``BlockManager.compute_block_hashes``: pure block hashing that neither
  reads nor writes per-request state and chains onto existing hashes.
- ``BlockManager.count_cached_prefix_blocks`` and the connectors'
  ``count_cached_prefix``: contiguous, tier-ordered (device -> host -> disk)
  hit counting with no side effects on pools, LRUs, or request state.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import cast

import numpy as np
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.connectors.null_connector import NullConnector
from max.pipelines.kv_cache.connectors.tiered_connector import TieredConnector
from max.pipelines.kv_cache.memory_tier import MemoryTier
from max.pipelines.kv_cache.paged_kv_cache.block_manager import (
    BlockManager,
    PrefixCacheHits,
)
from max.pipelines.kv_cache.paged_kv_cache.block_pool import BlockPool
from max.pipelines.modeling.types import RequestID

BLOCK_SIZE = 8


def _make_ctx(
    tokens: np.ndarray,
    request_id: RequestID = RequestID("req-1"),
) -> TextContext:
    """Build a minimal TextContext-like stub (see test_block_manager_sha256).

    ``compute_block_hashes`` reads ``ctx.pending_future_count`` (trailing
    future-token placeholders are excluded from hashing); the real
    ``TextContext`` always defines it (defaults to 0), so the stub must too.
    """
    ctx = SimpleNamespace(
        request_id=request_id,
        tokens=tokens,
        cache_salt=None,
        pending_future_count=0,
    )
    return cast(TextContext, ctx)


def _make_block_manager(
    *,
    connector: object | None = None,
    enable_prefix_caching: bool = True,
) -> BlockManager:
    return BlockManager(
        device_memory_tier=MemoryTier.MEMORY_TIER_CPU,
        total_num_blocks=32,
        block_size=BLOCK_SIZE,
        connector=cast(object, connector or NullConnector()),  # type: ignore[arg-type]
        enable_prefix_caching=enable_prefix_caching,
    )


def _seed_device_prefix_cache(
    bm: BlockManager, hashes: Sequence[bytes]
) -> None:
    """Commit blocks with the given hashes into the device prefix cache."""
    for h in hashes:
        block, _ = bm.device_block_pool.alloc_block()
        bm.device_block_pool.commit_into_prefix_cache(h, block)


class _TierStubConnector:
    """KVConnector-shaped stub with host/disk membership sets.

    Exercises the BlockManager -> connector hand-off: records the hashes it
    receives so tests can assert the walk starts where the device prefix
    ended, and asserts the canonical bytes form crossing the boundary.
    """

    def __init__(
        self,
        host_hashes: set[bytes] | None = None,
        disk_hashes: set[bytes] | None = None,
    ) -> None:
        self._host_hashes = host_hashes or set()
        self._disk_hashes = disk_hashes or set()
        self.received_hashes: list[bytes] | None = None

    @property
    def name(self) -> str:
        return "TierStubConnector"

    @property
    def num_host_blocks(self) -> int:
        return 4

    @property
    def supported_hash_algos(self) -> frozenset[str]:
        return frozenset({"ahash64", "sha256", "sha256_64"})

    def count_cached_prefix(
        self, block_hashes: Sequence[bytes]
    ) -> tuple[int, int]:
        assert all(isinstance(h, bytes) for h in block_hashes)
        self.received_hashes = list(block_hashes)
        num_host_hits = 0
        num_disk_hits = 0
        for h in block_hashes:
            if h in self._host_hashes:
                num_host_hits += 1
            elif h in self._disk_hashes:
                num_disk_hits += 1
            else:
                break
        return (num_host_hits, num_disk_hits)

    def load(
        self,
        device_block_ids: list[int],
        block_hashes: Sequence[bytes],
    ) -> int:
        raise NotImplementedError("must not be called by count paths")

    def offload(
        self,
        block_ids: list[int],
        block_hashes: Sequence[bytes],
        parent_seq_hash: bytes | None = None,
    ) -> None:
        raise NotImplementedError("must not be called by count paths")


# ---------------------------------------------------------------------------
# compute_block_hashes: pure hashing
# ---------------------------------------------------------------------------


def test_compute_block_hashes_is_side_effect_free() -> None:
    bm = _make_block_manager()
    # 33 tokens => 32 hashable (last reserved) => 4 full blocks of 8.
    ctx = _make_ctx(np.arange(33, dtype=np.int32))

    hashes = bm.compute_block_hashes(ctx, [])

    assert len(hashes) == 4
    assert ctx.request_id not in bm.req_to_hashes
    assert ctx.request_id not in bm.req_to_blocks
    assert ctx.request_id not in bm.req_to_committed_idx


def test_compute_block_hashes_matches_stateful_path() -> None:
    tokens = np.arange(33, dtype=np.int32)

    bm = _make_block_manager()
    pure = bm.compute_block_hashes(_make_ctx(tokens, RequestID("req-A")), [])

    bm.compute_hashes_for_request(_make_ctx(tokens, RequestID("req-B")))
    stateful = bm.req_to_hashes[RequestID("req-B")]

    assert pure == stateful


def test_compute_block_hashes_chains_onto_existing() -> None:
    """Hashes computed incrementally chain to the same values as one shot."""
    tokens = np.arange(33, dtype=np.int32)
    bm = _make_block_manager()
    ctx = _make_ctx(tokens)

    full = bm.compute_block_hashes(ctx, [])
    continuation = bm.compute_block_hashes(ctx, full[:2])

    assert continuation == full[2:]


def test_compute_block_hashes_partial_block_returns_empty() -> None:
    bm = _make_block_manager()
    # 8 tokens => 7 hashable => no full block of 8.
    ctx = _make_ctx(np.arange(8, dtype=np.int32))

    assert bm.compute_block_hashes(ctx, []) == []


# ---------------------------------------------------------------------------
# count_cached_prefix_blocks: device tier
# ---------------------------------------------------------------------------


def test_count_device_hits_contiguous_prefix() -> None:
    bm = _make_block_manager()
    ctx = _make_ctx(np.arange(33, dtype=np.int32))
    hashes = bm.compute_block_hashes(ctx, [])

    _seed_device_prefix_cache(bm, hashes[:3])

    hits = bm.count_cached_prefix_blocks(hashes)
    assert hits == PrefixCacheHits(device_blocks=3)
    assert hits.total_blocks == 3


def test_count_stops_at_device_gap() -> None:
    bm = _make_block_manager()
    ctx = _make_ctx(np.arange(33, dtype=np.int32))
    hashes = bm.compute_block_hashes(ctx, [])

    # Seed blocks 0 and 2, leaving a gap at block 1. With a NullConnector
    # (no external tiers) the walk must stop at the gap: block 2 is cached
    # but not reachable as part of the contiguous prefix.
    _seed_device_prefix_cache(bm, [hashes[0], hashes[2]])

    hits = bm.count_cached_prefix_blocks(hashes)
    assert hits == PrefixCacheHits(device_blocks=1)


def test_count_with_no_hits_is_zero() -> None:
    bm = _make_block_manager()
    ctx = _make_ctx(np.arange(33, dtype=np.int32))
    hashes = bm.compute_block_hashes(ctx, [])

    assert bm.count_cached_prefix_blocks(hashes) == PrefixCacheHits()


def test_count_respects_prefix_caching_disabled() -> None:
    bm = _make_block_manager(enable_prefix_caching=False)
    ctx = _make_ctx(np.arange(33, dtype=np.int32))
    hashes = bm.compute_block_hashes(ctx, [])

    assert bm.count_cached_prefix_blocks(hashes) == PrefixCacheHits()


def test_count_is_read_only() -> None:
    bm = _make_block_manager()
    ctx = _make_ctx(np.arange(33, dtype=np.int32))
    hashes = bm.compute_block_hashes(ctx, [])
    _seed_device_prefix_cache(bm, hashes[:2])

    num_free_before = bm.device_block_pool.num_free_blocks
    prefix_cache_before = dict(bm.device_block_pool.prefix_cache)

    first = bm.count_cached_prefix_blocks(hashes)
    second = bm.count_cached_prefix_blocks(hashes)

    assert first == second == PrefixCacheHits(device_blocks=2)
    assert bm.device_block_pool.num_free_blocks == num_free_before
    assert bm.device_block_pool.prefix_cache == prefix_cache_before
    assert ctx.request_id not in bm.req_to_hashes


# ---------------------------------------------------------------------------
# count_cached_prefix_blocks: continuation into connector tiers
# ---------------------------------------------------------------------------


def test_count_continues_into_host_and_disk_tiers() -> None:
    connector = _TierStubConnector()
    bm = _make_block_manager(connector=connector)
    ctx = _make_ctx(np.arange(41, dtype=np.int32))  # 5 full blocks
    hashes = bm.compute_block_hashes(ctx, [])

    # Block 0 on device, block 1 on host, block 2 on disk, block 3 missing,
    # block 4 on host (unreachable past the gap).
    _seed_device_prefix_cache(bm, hashes[:1])
    connector._host_hashes = {hashes[1], hashes[4]}
    connector._disk_hashes = {hashes[2]}

    hits = bm.count_cached_prefix_blocks(hashes)

    assert hits == PrefixCacheHits(
        device_blocks=1, host_blocks=1, disk_blocks=1
    )
    assert hits.total_blocks == 3
    # The connector must only be asked about the run after the device prefix.
    assert connector.received_hashes == list(hashes[1:])


def test_count_all_device_hits_skips_connector() -> None:
    connector = _TierStubConnector()
    bm = _make_block_manager(connector=connector)
    ctx = _make_ctx(np.arange(33, dtype=np.int32))
    hashes = bm.compute_block_hashes(ctx, [])
    _seed_device_prefix_cache(bm, hashes)

    hits = bm.count_cached_prefix_blocks(hashes)

    assert hits == PrefixCacheHits(device_blocks=4)
    assert connector.received_hashes is None


# ---------------------------------------------------------------------------
# Connector implementations
# ---------------------------------------------------------------------------


def test_null_connector_counts_nothing() -> None:
    assert NullConnector().count_cached_prefix([b"\x00" * 8]) == (0, 0)


def _make_tiered_connector_stub(
    host_hashes: Sequence[bytes],
    disk_hashes: Sequence[bytes],
    *,
    only_last_level: bool = False,
) -> TieredConnector:
    """Build a TieredConnector without device buffers.

    ``count_cached_prefix`` only touches the host block pool, the disk
    tier's ``contains``, and the last-level-cache flag, so the test wires
    exactly those three attributes onto an uninitialized instance.
    """
    connector = TieredConnector.__new__(TieredConnector)
    host_pool = BlockPool(
        MemoryTier.MEMORY_TIER_CPU,
        total_num_blocks=8,
        enable_prefix_caching=True,
        enable_runtime_checks=False,
    )
    for h in host_hashes:
        block, _ = host_pool.alloc_block()
        host_pool.commit_into_prefix_cache(h, block)
    connector._host_block_pool = host_pool
    disk_set = set(disk_hashes)
    connector._disk_tier = SimpleNamespace(contains=disk_set.__contains__)  # type: ignore[assignment]
    connector._only_use_kv_connector_last_level_cache = only_last_level
    return connector


def test_tiered_connector_walks_host_then_disk() -> None:
    h = [bytes([i]) * 8 for i in range(4)]
    connector = _make_tiered_connector_stub(
        host_hashes=[h[0]], disk_hashes=[h[1]]
    )

    # host, disk, miss, (unreached) => (1, 1)
    assert connector.count_cached_prefix(h) == (1, 1)


def test_tiered_connector_prefers_host_over_disk() -> None:
    h = [bytes([i]) * 8 for i in range(2)]
    connector = _make_tiered_connector_stub(host_hashes=h, disk_hashes=h)

    assert connector.count_cached_prefix(h) == (2, 0)


def test_tiered_connector_stops_at_gap() -> None:
    h = [bytes([i]) * 8 for i in range(3)]
    connector = _make_tiered_connector_stub(
        host_hashes=[h[0], h[2]], disk_hashes=[]
    )

    assert connector.count_cached_prefix(h) == (1, 0)


def test_tiered_connector_last_level_only_skips_host() -> None:
    h = [bytes([i]) * 8 for i in range(2)]
    connector = _make_tiered_connector_stub(
        host_hashes=[h[0], h[1]],
        disk_hashes=[h[0]],
        only_last_level=True,
    )

    # Host is ignored: block 0 counts from disk, block 1 is a miss.
    assert connector.count_cached_prefix(h) == (0, 1)
