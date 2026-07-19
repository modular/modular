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

"""Unit tests for ``BlockManager.offload`` sequence delivery.

These run CPU-only and construct a ``BlockManager`` directly with a recording
connector — no graph, session, or device memory. They cover the subtle parts of
delivering committed blocks as ordered parented offload sequences: hash
re-resolution to current device blocks, truncation of a run at the first block
evicted since commit (so the connector never sees a gap-chain), parent
pass-through, multi-run ordering, and that the pending queue is drained.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import cast

from max.nn.kv_cache import KVHashAlgo
from max.nn.kv_cache.cache_params import KVCacheMemory
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.kv_connector import to_block_hash_bytes
from max.pipelines.kv_cache.memory_tier import MemoryTier
from max.pipelines.kv_cache.paged_kv_cache.block_manager import BlockManager
from max.pipelines.kv_cache.paged_kv_cache.block_pool import BlockPool
from max.pipelines.kv_cache.paged_kv_cache.block_utils import KVCacheBlock
from max.pipelines.modeling.types import RequestID

# Short alias for readability in test fixtures and assertions: maps an int
# block hash to its canonical 8-byte big-endian signed encoding.
_b = to_block_hash_bytes


class RecordingConnector:
    """Connector stub that records ``offload``, ``touch`` and ``load`` calls."""

    def __init__(self) -> None:
        self.offloads: list[tuple[list[int], list[bytes], bytes | None]] = []
        self.touches: list[tuple[list[bytes], int]] = []
        # Ordered log of ``load``/``touch`` call names, so a test can assert the
        # load-path anchor touch fires AFTER the load (CLIN-1533).
        self.calls: list[str] = []
        # Blocks ``load`` reports as loaded from the host tier (0 == host miss);
        # lets a test drive a cold-G0/warm-host hit without real device memory.
        self.num_blocks_to_load = 0
        self._h2d_blocks_copied = 0
        self._d2h_blocks_copied = 0

    @property
    def name(self) -> str:
        return "recording"

    @property
    def supported_hash_algos(self) -> frozenset[KVHashAlgo]:
        return frozenset({"ahash64", "sha256", "sha256_64"})

    def offload(
        self,
        block_ids: list[int],
        block_hashes: Sequence[bytes],
        parent_seq_hash: bytes | None = None,
        replica_idx: int = 0,
    ) -> None:
        self.offloads.append((block_ids, list(block_hashes), parent_seq_hash))

    def touch(
        self,
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> None:
        self.calls.append("touch")
        self.touches.append((list(block_hashes), replica_idx))

    def load(
        self,
        device_block_ids: list[int],
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> int:
        self.calls.append("load")
        return min(len(block_hashes), self.num_blocks_to_load)

    def count_cached_prefix(
        self, block_hashes: Sequence[bytes]
    ) -> tuple[int, int]:
        return (0, 0)

    def wait_for_loads(self) -> None: ...
    def wait_for_offloads(self) -> None: ...
    def shutdown(self) -> None: ...
    def reset_prefix_cache(self) -> None: ...

    @property
    def num_host_blocks(self) -> int:
        return 0

    @property
    def num_used_host_blocks(self) -> int:
        return 0

    @property
    def num_disk_blocks(self) -> int:
        return 0

    @property
    def num_used_disk_blocks(self) -> int:
        return 0

    @property
    def metrics(self) -> KVCacheMetrics:
        return KVCacheMetrics(
            h2d_blocks_copied=self._h2d_blocks_copied,
            d2h_blocks_copied=self._d2h_blocks_copied,
        )

    def reset_metrics(self) -> None:
        self._h2d_blocks_copied = 0
        self._d2h_blocks_copied = 0


class _ExternalTierConnector(RecordingConnector):
    """A dKV-style connector that advertises an external tier.

    ``get_full_blocks_from_prefix_cache`` gates the G0 recency ``touch`` behind
    its ``num_host_blocks == 0`` early-return, so the touch-firing tests need a
    connector whose ``num_host_blocks`` is positive (a plain
    ``RecordingConnector`` reports 0, i.e. no external tier). Records touches
    like its base so a test can assert on them.
    """

    @property
    def num_host_blocks(self) -> int:
        return 1024


class _FakeKVMemory:
    """CPU stand-in for a ``KVCacheMemory`` unit: records cross-replica D2D
    copies without touching device memory, so a cross-replica prefix-cache hit
    can be exercised CPU-only."""

    def __init__(self) -> None:
        self.copies: list[tuple[int, int]] = []

    def copy_block_to(
        self, dst_unit: object, dst_block_id: int, src_block_id: int
    ) -> None:
        self.copies.append((dst_block_id, src_block_id))


def _make_block_manager(
    *,
    num_replicas: int = 1,
    connector: RecordingConnector | None = None,
    enable_dp_cross_replica_prefix_copy: bool = True,
) -> tuple[BlockManager, RecordingConnector]:
    connector = connector if connector is not None else RecordingConnector()
    # Multi-replica needs per-replica memory units so a cross-replica hit can
    # materialize via device-to-device copy (fake, CPU-only).
    replica_kv_memory = (
        cast(
            "Sequence[Sequence[KVCacheMemory]]",
            [[_FakeKVMemory()] for _ in range(num_replicas)],
        )
        if num_replicas > 1
        else None
    )
    bm = BlockManager(
        device_memory_tier=MemoryTier.MEMORY_TIER_CPU,
        total_num_blocks=64,
        block_size=16,
        connector=connector,
        enable_prefix_caching=True,
        num_replicas=num_replicas,
        replica_kv_memory=replica_kv_memory,
        enable_dp_cross_replica_prefix_copy=enable_dp_cross_replica_prefix_copy,
    )
    return bm, connector


def _commit(bm: BlockManager, hash_to_bid: dict[bytes, int]) -> None:
    """Place ``hash -> KVCacheBlock(bid)`` entries in the device prefix cache."""
    for block_hash, bid in hash_to_bid.items():
        bm.device_block_pool.prefix_cache[block_hash] = KVCacheBlock(bid)


def _make_ctx(request_id: RequestID) -> TextContext:
    """Minimal ctx stub.

    ``get_full_blocks_from_prefix_cache`` reads only ``ctx.request_id`` on this
    path (no tokens/salt/images), so a ``SimpleNamespace`` suffices.
    """
    return cast(TextContext, SimpleNamespace(request_id=request_id))


def _commit_device_block(pool: BlockPool, block_hash: int) -> KVCacheBlock:
    """Commit ``block_hash`` as an idle eviction-candidate device block.

    Unlike :func:`_commit` (which injects a bare ``KVCacheBlock`` used only by
    the offload path), this allocates, commits, then frees a real pool block so
    it sits in both the prefix cache and the free queue at ``ref_cnt == 0`` --
    the realistic state a device prefix-cache *hit* resolves to, and the state
    that lets the hit's ``BlockPool.touch`` exercise the free-queue path without
    corrupting it.
    """
    block, _ = pool.alloc_block()
    pool.commit_into_prefix_cache(block_hash, block)
    pool.free_block(block)
    return block


def test_offload_delivers_run_resolving_hashes_to_bids() -> None:
    bm, connector = _make_block_manager()
    _commit(bm, {_b(111): 5, _b(222): 6, _b(333): 7})
    # One run of three committed blocks chaining onto parent 999.
    bm._pending_offloads = [[(_b(999), [_b(111), _b(222), _b(333)])]]

    bm.offload()

    assert connector.offloads == [
        ([5, 6, 7], [_b(111), _b(222), _b(333)], _b(999))
    ]
    # Pending queue drained.
    assert bm._pending_offloads == [[]]


def test_offload_root_run_uses_parent_none() -> None:
    bm, connector = _make_block_manager()
    _commit(bm, {_b(111): 5, _b(222): 6})
    bm._pending_offloads = [[(None, [_b(111), _b(222)])]]

    bm.offload()

    assert connector.offloads == [([5, 6], [_b(111), _b(222)], None)]


def test_offload_truncates_run_at_evicted_block() -> None:
    bm, connector = _make_block_manager()
    # 222 was evicted since commit; the run must stop before it so the chain
    # has no gap (333's parent would otherwise be missing).
    _commit(bm, {_b(111): 5, _b(333): 7})
    bm._pending_offloads = [[(None, [_b(111), _b(222), _b(333)])]]

    bm.offload()

    assert connector.offloads == [([5], [_b(111)], None)]


def test_offload_skips_fully_evicted_run() -> None:
    bm, connector = _make_block_manager()
    # First (and only) block of the run is gone -> nothing to deliver.
    _commit(bm, {})
    bm._pending_offloads = [[(None, [_b(111)])]]

    bm.offload()

    assert connector.offloads == []
    assert bm._pending_offloads == [[]]


def test_offload_preserves_multi_run_order() -> None:
    bm, connector = _make_block_manager()
    _commit(bm, {_b(111): 1, _b(222): 2, _b(333): 3, _b(444): 4})
    # Two runs queued across two commits; second chains onto the first's tail.
    bm._pending_offloads = [
        [
            (None, [_b(111), _b(222)]),
            (_b(222), [_b(333), _b(444)]),
        ]
    ]

    bm.offload()

    assert connector.offloads == [
        ([1, 2], [_b(111), _b(222)], None),
        ([3, 4], [_b(333), _b(444)], _b(222)),
    ]


def test_reset_metrics_clears_connector_transfer_counters() -> None:
    """Per-batch telemetry must reset connector H2D/D2H counters after sampling.

    Without this, ``get_metrics_aggregated()`` returns lifetime cumulative
    totals and Datadog counter.add() double-counts across batches (MXSERV-203).
    """
    bm, connector = _make_block_manager()
    connector._d2h_blocks_copied = 5
    connector._h2d_blocks_copied = 2

    assert bm.metrics.d2h_blocks_copied == 5
    assert bm.metrics.h2d_blocks_copied == 2

    bm.reset_metrics()

    assert bm.metrics.d2h_blocks_copied == 0
    assert bm.metrics.h2d_blocks_copied == 0

    connector._d2h_blocks_copied = 3
    assert bm.metrics.d2h_blocks_copied == 3
    assert bm.metrics.h2d_blocks_copied == 0


def test_touch_fires_on_device_hit_with_full_root_anchored_hashes() -> None:
    """A G0 device prefix-cache hit touches the FULL root-anchored sequence.

    The request has a 4-block root-anchored prefix whose first two blocks are
    already committed (``num_committed_blocks == 2``), so the device is queried
    for the root-omitting slice ``[333, 444]``. The touch payload must still be
    the full ``[111, 222, 333, 444]`` -- not that slice -- so the prefix root
    stays MRU under dKV's reverse full-attention LRU (the ordering correction,
    CLIN-1533). It fires exactly once and, the whole prefix being on device,
    issues no ``load``. Uses an external-tier connector because the anchor is
    gated on ``num_host_blocks``.
    """
    bm, connector = _make_block_manager(connector=_ExternalTierConnector())
    rid = RequestID("req-hit")
    bm.req_to_hashes[rid] = [111, 222, 333, 444]
    # First two blocks already committed => num_committed_blocks == 2.
    bm.req_to_committed_idx[rid] = 2 * bm.block_size
    _commit_device_block(bm.device_block_pool, 333)
    _commit_device_block(bm.device_block_pool, 444)

    device_blocks = bm.get_full_blocks_from_prefix_cache(_make_ctx(rid))

    assert len(device_blocks) == 2  # the two uncommitted device hits
    assert connector.calls == ["touch"]  # fires once; no host load
    assert connector.touches == [([_b(111), _b(222), _b(333), _b(444)], 0)]


def test_touch_anchor_not_fired_on_fully_cold_request() -> None:
    """Fully cold (no device hit AND no host hit) means no anchor touch.

    Nothing is resident on device and the host tier loads nothing
    (``num_blocks_to_load == 0``), so both ``device_blocks`` and ``host_blocks``
    are empty and the ``if device_blocks or host_blocks`` gate suppresses the
    anchor -- even though the load path ran (``load`` was called). Uses an
    external-tier connector so the ``num_host_blocks`` gate is passed and the
    empty-result gate is what's exercised.
    """
    bm, connector = _make_block_manager(connector=_ExternalTierConnector())
    rid = RequestID("req-cold")
    bm.req_to_hashes[rid] = [111, 222]  # nothing on device, nothing in host

    served = bm.get_full_blocks_from_prefix_cache(_make_ctx(rid))

    assert served == []  # nothing served
    assert connector.calls == ["load"]  # load ran; gate suppressed the touch
    assert connector.touches == []


def test_touch_fires_on_cross_replica_hit_keyed_to_serving_replica() -> None:
    """A cross-replica device hit still touches the full sequence.

    The blocks are resident only on replica 1's prefix cache; the request runs
    on replica 0, so the hit is served by a device-to-device materialization
    onto replica 0. The touch carries the full root-anchored sequence and is
    keyed to the *serving* replica (0, the client selector) -- not the source
    replica (1).
    """
    bm, connector = _make_block_manager(
        num_replicas=2, connector=_ExternalTierConnector()
    )
    rid = RequestID("req-xrep")
    bm.req_to_hashes[rid] = [111, 222]
    _commit_device_block(bm.device_block_pools[1], 111)
    _commit_device_block(bm.device_block_pools[1], 222)

    device_blocks = bm.get_full_blocks_from_prefix_cache(
        _make_ctx(rid), replica_idx=0
    )

    assert len(device_blocks) == 2  # materialized onto replica 0
    assert connector.touches == [([_b(111), _b(222)], 0)]


def test_cross_replica_copy_disabled_serves_from_external_tier() -> None:
    """With enable_dp_cross_replica_prefix_copy off, a block resident only on
    another replica's device is NOT materialized via a device-to-device copy:
    the device lookup stops at the local miss and the prefix is served from
    the shared external tier instead.
    """
    connector = _ExternalTierConnector()
    connector.num_blocks_to_load = 2  # both blocks are warm in the tier
    bm, _ = _make_block_manager(
        num_replicas=2,
        connector=connector,
        enable_dp_cross_replica_prefix_copy=False,
    )
    rid = RequestID("req-xrep-off")
    bm.req_to_hashes[rid] = [111, 222]
    _commit_device_block(bm.device_block_pools[1], 111)
    _commit_device_block(bm.device_block_pools[1], 222)

    served = bm.get_full_blocks_from_prefix_cache(_make_ctx(rid), replica_idx=0)

    assert len(served) == 2  # served, but by the external tier
    assert connector.calls == ["load", "touch"]  # host load, no device hit
    assert bm.metrics.cross_replica_blocks_copied == 0
    assert bm._replica_kv_memory is not None
    for units in bm._replica_kv_memory:
        assert cast(_FakeKVMemory, units[0]).copies == []  # no D2D issued


def test_cross_replica_copy_disabled_count_is_local_only() -> None:
    """With the flag off, the admission estimate must not count blocks the
    reuse path can no longer serve by device-to-device copy: only the request
    replica's own resident blocks count as device hits.
    """
    bm, _ = _make_block_manager(
        num_replicas=2, enable_dp_cross_replica_prefix_copy=False
    )
    _commit_device_block(bm.device_block_pools[1], 111)
    _commit_device_block(bm.device_block_pools[1], 222)

    assert (
        bm._count_full_blocks_from_prefix_cache([111, 222], replica_idx=0) == 0
    )
    assert (
        bm._count_full_blocks_from_prefix_cache([111, 222], replica_idx=1) == 2
    )


def test_touch_not_fired_without_external_tier() -> None:
    """A connector with no external tier is never touched on a device hit.

    The G0 recency ``touch`` is gated behind the ``num_host_blocks == 0``
    early-return, so a NullConnector-style connector (``num_host_blocks == 0``,
    whose ``touch`` is a pure no-op) sees no ``touch`` call at all and pays no
    per-hit payload cost. A plain ``RecordingConnector`` reports
    ``num_host_blocks == 0`` and records any touch, so the empty ``touches``
    assertion verifies the gate rather than a silent no-op.
    """
    bm, connector = (
        _make_block_manager()
    )  # RecordingConnector: no external tier
    rid = RequestID("req-null")
    bm.req_to_hashes[rid] = [111, 222]
    _commit_device_block(bm.device_block_pool, 111)
    _commit_device_block(bm.device_block_pool, 222)

    device_blocks = bm.get_full_blocks_from_prefix_cache(_make_ctx(rid))

    assert len(device_blocks) == 2
    assert connector.touches == []


def test_touch_anchor_fires_after_load_on_host_only_hit() -> None:
    """Cold-G0 / warm-external hit: the anchor fires AFTER the load, once.

    Nothing is resident on device, so ``device_blocks`` is empty and the whole
    prefix is pulled from the external tier by the load. The anchor is the SOLE
    load-path recency signal, so it must fire AFTER that load -- as the only
    load-path toucher it thereby reserves the last recency stamp and cannot be
    inverted by the load's own touches (CLIN-1533). It fires exactly once, with
    the root-anchored payload, even though there was no device hit.
    """
    connector = _ExternalTierConnector()
    connector.num_blocks_to_load = 2  # both requested blocks load from host
    bm, _ = _make_block_manager(connector=connector)
    rid = RequestID("req-host")
    bm.req_to_hashes[rid] = [111, 222]  # nothing committed, nothing on device

    served = bm.get_full_blocks_from_prefix_cache(_make_ctx(rid))

    assert len(served) == 2  # both served from the host tier (no device hit)
    assert connector.calls == ["load", "touch"]  # touch after load, once
    assert connector.touches == [([_b(111), _b(222)], 0)]


def test_touch_anchor_payload_trims_uncached_tail() -> None:
    """Anchor payload is root-anchored: committed prefix in, uncached tail out.

    A 3-block request whose first block is already committed
    (``num_committed_blocks == 1``) hits the device for block 2 only; block 3
    is uncached (absent on device and in the host tier). The touch payload
    must be ``[111, 222]``: it INCLUDES the committed root 111 (omitting it
    re-creates a recency inversion) and EXCLUDES the uncached tail 333 (absent
    server-side, so touching it is a wasted index lookup on a contended path).
    """
    bm, connector = _make_block_manager(connector=_ExternalTierConnector())
    rid = RequestID("req-tail")
    bm.req_to_hashes[rid] = [111, 222, 333]
    # First block already committed => num_committed_blocks == 1.
    bm.req_to_committed_idx[rid] = 1 * bm.block_size
    _commit_device_block(bm.device_block_pool, 222)  # 333 stays uncached

    served = bm.get_full_blocks_from_prefix_cache(_make_ctx(rid))

    assert len(served) == 1  # only 222 hit; 333 is the uncached tail
    assert connector.touches == [([_b(111), _b(222)], 0)]  # root in, tail out
