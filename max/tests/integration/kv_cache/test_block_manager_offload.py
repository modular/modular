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

from max.nn.kv_cache import KVHashAlgo
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.kv_cache.kv_connector import to_block_hash_bytes
from max.pipelines.kv_cache.memory_tier import MemoryTier
from max.pipelines.kv_cache.paged_kv_cache.block_manager import BlockManager
from max.pipelines.kv_cache.paged_kv_cache.block_utils import KVCacheBlock

# Short alias for readability in test fixtures and assertions: maps an int
# block hash to its canonical 8-byte big-endian signed encoding.
_b = to_block_hash_bytes


class RecordingConnector:
    """Connector stub that records the ``offload`` calls it receives."""

    def __init__(self) -> None:
        self.offloads: list[tuple[list[int], list[bytes], bytes | None]] = []

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

    def load(
        self,
        device_block_ids: list[int],
        block_hashes: Sequence[bytes],
        replica_idx: int = 0,
    ) -> int:
        return 0

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
        return KVCacheMetrics()


def _make_block_manager() -> tuple[BlockManager, RecordingConnector]:
    connector = RecordingConnector()
    bm = BlockManager(
        device_memory_tier=MemoryTier.MEMORY_TIER_CPU,
        total_num_blocks=64,
        block_size=16,
        connector=connector,
        enable_prefix_caching=True,
    )
    return bm, connector


def _commit(bm: BlockManager, hash_to_bid: dict[bytes, int]) -> None:
    """Place ``hash -> KVCacheBlock(bid)`` entries in the device prefix cache."""
    for block_hash, bid in hash_to_bid.items():
        bm.device_block_pool.prefix_cache[block_hash] = KVCacheBlock(bid)


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
