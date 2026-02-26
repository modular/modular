# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

"""GPU integration tests for TieredConnector (GPU <-> CPU <-> Disk)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.graph import DeviceRef
from max.kv_cache.connectors.tiered_connector import TieredConnector
from max.nn.kv_cache import KVCacheBuffer, KVCacheParams
from test_common.context_utils import create_text_context


def create_tiered_connector(
    num_device_blocks: int = 64,
    num_host_blocks: int = 32,
    page_size: int = 16,
    num_layers: int = 2,
    n_kv_heads: int = 4,
    head_dim: int = 64,
    disk_cache_dir: str | None = None,
    max_disk_size_gb: float = 1.0,
) -> TieredConnector:
    """Create a TieredConnector for testing."""
    if accelerator_count() == 0:
        pytest.skip("No GPU available")

    if disk_cache_dir is None:
        disk_cache_dir = tempfile.mkdtemp(prefix="tiered_test_")

    device = Accelerator()
    kv_params = KVCacheParams(
        dtype=DType.float32,
        num_layers=num_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        cache_strategy="paged",
        enable_prefix_caching=True,
        enable_kvcache_swapping_to_host=True,
        host_kvcache_swap_space_gb=999,
        page_size=page_size,
        devices=[DeviceRef.GPU()],
    )

    device_buffers = [
        Buffer(
            shape=[num_device_blocks, *kv_params.shape_per_block],
            dtype=kv_params.dtype,
            device=device,
        )
    ]

    return TieredConnector(
        params=kv_params,
        devices=[device],
        device_buffer=KVCacheBuffer(
            total_num_pages=num_device_blocks, values=device_buffers
        ),
        total_num_host_blocks=num_host_blocks,
        disk_cache_dir=disk_cache_dir,
        max_disk_size_gb=max_disk_size_gb,
    )


# -- Basic properties --


def test_connector_name() -> None:
    connector = create_tiered_connector()
    assert connector.name == "TieredConnector"
    connector.shutdown()


def test_host_tensors_are_pinned() -> None:
    connector = create_tiered_connector()
    assert connector._host_buffer
    for tensor in connector._host_buffer.all_buffers:
        assert tensor.pinned, "Host tensors should be pinned memory"
    connector.shutdown()


def test_num_host_blocks() -> None:
    connector = create_tiered_connector(num_host_blocks=48)
    assert connector.num_host_blocks == 48
    connector.shutdown()


def test_num_used_host_blocks_initially_zero() -> None:
    connector = create_tiered_connector()
    assert connector.num_used_host_blocks == 0
    connector.shutdown()


# -- Save / flush / sync --


def test_save_queues_blocks_for_offload() -> None:
    connector = create_tiered_connector()
    connector.save([0, 1, 2], [100, 200, 300])
    assert len(connector._pending_saves) == 3
    assert connector.num_used_host_blocks == 0
    connector.shutdown()


def test_flush_executes_pending_saves() -> None:
    connector = create_tiered_connector()
    connector.save([0, 1], [100, 200])
    connector.flush()
    assert len(connector._pending_saves) == 0
    assert connector.num_used_host_blocks == 2
    connector.shutdown()


def test_duplicate_hash_not_saved_twice() -> None:
    connector = create_tiered_connector()
    connector.save([0], [100])
    connector.flush()
    connector.save([1], [100])
    connector.flush()
    assert connector.num_used_host_blocks == 1
    connector.shutdown()


# -- Lookup / load (CPU hits) --


def test_lookup_returns_zero_for_empty_cache() -> None:
    connector = create_tiered_connector(page_size=16)
    ctx = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    tokens = connector.lookup(ctx, [100, 200, 300])
    assert tokens == 0
    connector.shutdown()


def test_lookup_finds_cached_blocks() -> None:
    page_size = 16
    connector = create_tiered_connector(page_size=page_size)
    connector.save([0, 1, 2], [100, 200, 300])
    connector.flush()

    ctx = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    tokens = connector.lookup(ctx, [100, 200, 300])
    assert tokens == 3 * page_size
    connector.shutdown()


def test_lookup_stops_at_first_miss() -> None:
    page_size = 16
    connector = create_tiered_connector(page_size=page_size)
    connector.save([0], [100])
    connector.save([2], [300])
    connector.flush()

    ctx = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    tokens = connector.lookup(ctx, [100, 200, 300])
    assert tokens == 1 * page_size
    connector.shutdown()


def test_load_returns_hashes_for_loaded_blocks() -> None:
    connector = create_tiered_connector(page_size=16)
    connector.save([0, 1], [100, 200])
    connector.flush()

    ctx = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    connector.lookup(ctx, [100, 200])
    loaded_hashes = connector.load(ctx, [10, 11])
    assert loaded_hashes == [100, 200]
    connector.shutdown()


def test_load_without_lookup_returns_empty() -> None:
    connector = create_tiered_connector()
    ctx = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    loaded_hashes = connector.load(ctx, [0, 1])
    assert loaded_hashes == []
    connector.shutdown()


# -- Write-through to disk --


def test_write_through_to_disk() -> None:
    """Verify blocks written to CPU are also written through to disk."""
    with tempfile.TemporaryDirectory(prefix="tiered_wt_") as disk_dir:
        connector = create_tiered_connector(disk_cache_dir=disk_dir)

        connector.save([0, 1, 2], [100, 200, 300])
        connector.flush()
        # flush() records pending disk writes; sync() executes them
        connector.sync()

        # Wait for async disk writes to complete
        connector._disk_tier.wait_for_writes()

        assert connector._disk_tier.contains(100)
        assert connector._disk_tier.contains(200)
        assert connector._disk_tier.contains(300)

        # Verify files exist on disk
        bin_files = list(Path(disk_dir).glob("*.bin"))
        assert len(bin_files) == 3

        connector.shutdown()


def test_write_through_skips_already_on_disk() -> None:
    """Verify write-through skips blocks already present on disk."""
    with tempfile.TemporaryDirectory(prefix="tiered_skip_") as disk_dir:
        connector = create_tiered_connector(disk_cache_dir=disk_dir)

        # First save
        connector.save([0], [100])
        connector.flush()
        connector.sync()
        connector._disk_tier.wait_for_writes()

        written_before = connector._disk_blocks_written

        # Save same hash again (from different device block)
        connector.save([1], [100])
        connector.flush()
        connector.sync()
        connector._disk_tier.wait_for_writes()

        # Should not have written again (deduplicated at CPU level)
        assert connector._disk_blocks_written == written_before

        connector.shutdown()


# -- Disk promotion (disk -> CPU -> GPU) --


def test_disk_promotion_to_cpu() -> None:
    """Verify lookup promotes blocks from disk to CPU when not in CPU cache."""
    with tempfile.TemporaryDirectory(prefix="tiered_promo_") as disk_dir:
        # Very small CPU cache (4 blocks) to force eviction
        connector = create_tiered_connector(
            num_host_blocks=4,
            disk_cache_dir=disk_dir,
        )
        page_size = 16

        # Save 4 blocks → fills CPU cache
        connector.save([0, 1, 2, 3], [100, 200, 300, 400])
        connector.flush()
        connector.sync()
        connector._disk_tier.wait_for_writes()
        connector.sync()  # drain write-locked blocks

        # All 4 should be on disk
        for h in [100, 200, 300, 400]:
            assert connector._disk_tier.contains(h)

        # Save 4 more blocks → evicts the first 4 from CPU
        connector.save([4, 5, 6, 7], [500, 600, 700, 800])
        connector.flush()
        connector.sync()
        connector._disk_tier.wait_for_writes()
        connector.sync()  # drain write-locked blocks

        # The first 4 should still be on disk (write-through)
        for h in [100, 200, 300, 400]:
            assert connector._disk_tier.contains(h)

        # Lookup hash 100 → should find it on disk and promote to CPU
        ctx = create_text_context(np.array([1], dtype=np.int64))
        tokens = connector.lookup(ctx, [100])
        assert tokens == page_size  # disk hit

        # Load should work (waits for disk read, then H2D)
        loaded = connector.load(ctx, [10])
        assert loaded == [100]

        connector.shutdown()


# -- Full round-trip --


def test_full_round_trip() -> None:
    """Save -> write-through -> evict from CPU -> lookup from disk -> load to GPU."""
    with tempfile.TemporaryDirectory(prefix="tiered_rt_") as disk_dir:
        connector = create_tiered_connector(
            num_host_blocks=2,
            disk_cache_dir=disk_dir,
        )
        page_size = 16

        # Save 2 blocks (fills CPU)
        connector.save([0, 1], [100, 200])
        connector.flush()
        connector.sync()
        connector._disk_tier.wait_for_writes()
        connector.sync()  # drain write-locked blocks

        # Save 2 more → evicts 100, 200 from CPU
        connector.save([2, 3], [300, 400])
        connector.flush()
        connector.sync()
        connector._disk_tier.wait_for_writes()
        connector.sync()  # drain write-locked blocks

        # 100, 200 should still be on disk
        assert connector._disk_tier.contains(100)
        assert connector._disk_tier.contains(200)

        # Lookup [100, 200] → both promoted from disk
        ctx = create_text_context(np.array([1, 2], dtype=np.int64))
        tokens = connector.lookup(ctx, [100, 200])
        assert tokens == 2 * page_size

        # Load to GPU
        loaded = connector.load(ctx, [20, 21])
        assert loaded == [100, 200]

        connector.shutdown()


# -- Metrics --


def test_metrics_track_disk_operations() -> None:
    """Verify disk metrics are tracked correctly."""
    with tempfile.TemporaryDirectory(prefix="tiered_met_") as disk_dir:
        connector = create_tiered_connector(
            num_host_blocks=2,
            disk_cache_dir=disk_dir,
        )

        # Write 2 blocks through to disk
        connector.save([0, 1], [100, 200])
        connector.flush()
        connector.sync()
        connector._disk_tier.wait_for_writes()
        connector.sync()  # drain write-locked blocks

        metrics = connector.metrics
        assert metrics.d2h_blocks_copied == 2
        assert metrics.disk_blocks_written == 2

        # Evict from CPU, then promote from disk
        connector.save([2, 3], [300, 400])
        connector.flush()
        connector.sync()
        connector._disk_tier.wait_for_writes()
        connector.sync()  # drain write-locked blocks

        ctx = create_text_context(np.array([1], dtype=np.int64))
        connector.lookup(ctx, [100])
        connector.load(ctx, [10])

        metrics = connector.metrics
        assert metrics.disk_blocks_read >= 1
        assert metrics.h2d_blocks_copied >= 1

        connector.shutdown()


# -- Prefix chain with disk gap --


def test_lookup_breaks_chain_at_disk_miss() -> None:
    """Verify lookup stops at first complete miss (not on CPU or disk)."""
    with tempfile.TemporaryDirectory(prefix="tiered_gap_") as disk_dir:
        connector = create_tiered_connector(disk_cache_dir=disk_dir)
        page_size = 16

        # Only save hash 100 and 300, NOT 200
        connector.save([0], [100])
        connector.save([2], [300])
        connector.flush()
        connector.sync()
        connector._disk_tier.wait_for_writes()

        ctx = create_text_context(np.array([1, 2, 3], dtype=np.int64))
        # Lookup [100, 200, 300] → should stop at 200 (miss)
        tokens = connector.lookup(ctx, [100, 200, 300])
        assert tokens == 1 * page_size  # only hash 100

        connector.shutdown()


# -- Request lifecycle --


def test_on_request_complete_clears_state() -> None:
    connector = create_tiered_connector()
    connector.save([0], [100])
    connector.flush()

    ctx = create_text_context(np.array([1], dtype=np.int64))
    connector.lookup(ctx, [100])
    assert str(ctx.request_id) in connector._pending_loads

    connector.on_request_complete(ctx.request_id, [0])
    assert str(ctx.request_id) not in connector._pending_loads
    assert str(ctx.request_id) not in connector._pending_disk_reads

    connector.shutdown()


def test_shutdown_clears_pending_state() -> None:
    connector = create_tiered_connector()
    connector.save([0, 1], [100, 200])
    connector.shutdown()

    assert len(connector._pending_saves) == 0
    assert len(connector._pending_loads) == 0
    assert len(connector._pending_disk_writes) == 0
    assert len(connector._pending_disk_reads) == 0


# -- Reset --


def test_reset_prefix_cache_clears_cpu_and_disk() -> None:
    with tempfile.TemporaryDirectory(prefix="tiered_reset_") as disk_dir:
        connector = create_tiered_connector(disk_cache_dir=disk_dir)

        connector.save([0, 1], [100, 200])
        connector.flush()
        connector.sync()
        connector._disk_tier.wait_for_writes()
        assert connector.num_used_host_blocks == 2
        assert connector._disk_tier.contains(100)

        connector.reset_prefix_cache()

        assert connector.num_used_host_blocks == 0
        assert not connector._disk_tier.contains(100)
        assert not connector._disk_tier.contains(200)

        connector.shutdown()


# -- Warm restart --


def test_warm_restart_loads_disk_cache() -> None:
    """Verify a new TieredConnector finds blocks persisted by a previous one."""
    with tempfile.TemporaryDirectory(prefix="tiered_warm_") as disk_dir:
        # First connector: write blocks to disk
        c1 = create_tiered_connector(disk_cache_dir=disk_dir)
        c1.save([0, 1], [100, 200])
        c1.flush()
        c1.sync()
        c1._disk_tier.wait_for_writes()
        c1.shutdown()  # saves metadata

        # Second connector: same disk dir → should load metadata
        c2 = create_tiered_connector(disk_cache_dir=disk_dir)
        assert c2._disk_tier.contains(100)
        assert c2._disk_tier.contains(200)

        # Should be able to promote from disk
        page_size = 16
        ctx = create_text_context(np.array([1, 2], dtype=np.int64))
        tokens = c2.lookup(ctx, [100, 200])
        assert tokens == 2 * page_size

        loaded = c2.load(ctx, [10, 11])
        assert loaded == [100, 200]

        c2.shutdown()


# -- Eviction protection (Change 1: deferred free_block) --


def test_lookup_pins_blocks_during_disk_read() -> None:
    """Verify host blocks stay pinned (ref_cnt=1) after disk promotion in lookup().

    Before the fix, lookup() called free_block() immediately after
    commit_into_prefix_cache(), making the block evictable while the
    async disk read was still writing into its memory.  Now free_block()
    is deferred to load() so the block is protected from eviction.
    """
    with tempfile.TemporaryDirectory(prefix="tiered_pin_") as disk_dir:
        # Small CPU cache (4 blocks) to force eviction
        connector = create_tiered_connector(
            num_host_blocks=4,
            disk_cache_dir=disk_dir,
        )

        # Fill CPU cache and write through to disk
        connector.save([0, 1, 2, 3], [100, 200, 300, 400])
        connector.flush()
        connector.sync()
        connector._disk_tier.wait_for_writes()
        connector.sync()  # drain write-locked blocks

        # Evict blocks 100-400 from CPU by saving 4 new blocks
        connector.save([4, 5, 6, 7], [500, 600, 700, 800])
        connector.flush()
        connector.sync()
        connector._disk_tier.wait_for_writes()
        connector.sync()  # drain write-locked blocks

        # Now lookup hash 100 — should promote from disk
        ctx = create_text_context(np.array([1], dtype=np.int64))
        tokens = connector.lookup(ctx, [100])
        assert tokens == 16  # disk hit

        # The promoted block should be pinned (ref_cnt > 0) so it
        # can't be evicted while the async disk read completes.
        pending = connector._pending_loads[str(ctx.request_id)]
        for host_block, _hash in pending:
            assert host_block.ref_cnt > 0, (
                "Host block should be pinned after disk promotion"
            )

        # After load(), the block should be released (ref_cnt=0, in free queue)
        loaded = connector.load(ctx, [10])
        assert loaded == [100]
        for host_block, _hash in pending:
            assert host_block.ref_cnt == 0, (
                "Host block should be released after load()"
            )

        connector.shutdown()


def test_on_request_complete_releases_pinned_disk_promotion_blocks() -> None:
    """Verify on_request_complete frees blocks from lookup() that load() never consumed."""
    with tempfile.TemporaryDirectory(prefix="tiered_orc_") as disk_dir:
        connector = create_tiered_connector(
            num_host_blocks=4,
            disk_cache_dir=disk_dir,
        )

        # Fill CPU + disk, then evict from CPU
        connector.save([0, 1, 2, 3], [100, 200, 300, 400])
        connector.flush()
        connector.sync()
        connector._disk_tier.wait_for_writes()
        connector.sync()  # drain write-locked blocks

        connector.save([4, 5, 6, 7], [500, 600, 700, 800])
        connector.flush()
        connector.sync()
        connector._disk_tier.wait_for_writes()
        connector.sync()  # drain write-locked blocks

        free_before = (
            connector._host_block_pool.free_block_queue.num_free_blocks
        )

        # Lookup from disk (pins blocks) but DON'T call load()
        ctx = create_text_context(np.array([1], dtype=np.int64))
        tokens = connector.lookup(ctx, [100])
        assert tokens == 16

        # One block is pinned, so free count should decrease
        free_after_lookup = (
            connector._host_block_pool.free_block_queue.num_free_blocks
        )
        assert free_after_lookup < free_before

        # on_request_complete should release the pinned block
        connector.on_request_complete(ctx.request_id, [0])

        free_after_complete = (
            connector._host_block_pool.free_block_queue.num_free_blocks
        )
        assert free_after_complete == free_before, (
            "on_request_complete should release pinned blocks"
        )

        connector.shutdown()


# -- Zero-copy disk writes (Change 4) --


def test_write_locked_blocks_lifecycle() -> None:
    """Verify _write_locked_blocks tracks in-flight writes and drains correctly."""
    with tempfile.TemporaryDirectory(prefix="tiered_wlb_") as disk_dir:
        connector = create_tiered_connector(disk_cache_dir=disk_dir)

        # Save and flush → creates pending_disk_writes
        connector.save([0, 1], [100, 200])
        connector.flush()

        assert len(connector._pending_disk_writes) == 2

        # sync() submits writes to disk and tracks in _write_locked_blocks
        connector.sync()

        # After sync, _pending_disk_writes is cleared
        assert len(connector._pending_disk_writes) == 0

        # _write_locked_blocks may have entries (in-flight writes)
        # or they may have already completed
        locked_count = len(connector._write_locked_blocks)

        # Wait for all writes to complete
        connector._disk_tier.wait_for_writes()

        # Next drain should release all completed write blocks
        connector._drain_completed_writes()
        assert len(connector._write_locked_blocks) == 0, (
            "All write-locked blocks should be drained after writes complete"
        )

        connector.shutdown()


def test_host_blocks_pinned_during_disk_write() -> None:
    """Verify host blocks stay pinned (ref_cnt=1) while disk writes are in-flight.

    This is the zero-copy guarantee: host blocks are not freed until the
    disk write thread is done reading from their memory.
    """
    with tempfile.TemporaryDirectory(prefix="tiered_zc_") as disk_dir:
        connector = create_tiered_connector(disk_cache_dir=disk_dir)

        # Save and flush → D2H copies queued
        connector.save([0, 1], [100, 200])
        connector.flush()

        # At this point, host blocks should be pinned (ref_cnt=1)
        # because _maybe_offload_to_host() no longer calls free_block()
        for _bid, _hash, host_block in connector._pending_disk_writes:
            assert host_block.ref_cnt > 0, (
                "Host block should be pinned for disk write safety"
            )

        # After sync() + wait, blocks should eventually be released
        connector.sync()
        connector._disk_tier.wait_for_writes()
        connector._drain_completed_writes()

        assert len(connector._write_locked_blocks) == 0

        connector.shutdown()


def test_drain_completed_writes_across_sync_cycles() -> None:
    """Verify _drain_completed_writes releases blocks from previous cycles."""
    with tempfile.TemporaryDirectory(prefix="tiered_drain_") as disk_dir:
        connector = create_tiered_connector(disk_cache_dir=disk_dir)

        # Cycle 1: save + flush + sync
        connector.save([0, 1], [100, 200])
        connector.flush()
        connector.sync()

        # Wait for cycle 1 writes to complete
        connector._disk_tier.wait_for_writes()

        # Cycle 2: save + flush + sync — should drain cycle 1's blocks
        connector.save([2, 3], [300, 400])
        connector.flush()
        connector.sync()  # calls _drain_completed_writes() internally

        # Wait for cycle 2
        connector._disk_tier.wait_for_writes()
        connector._drain_completed_writes()

        assert len(connector._write_locked_blocks) == 0, (
            "All blocks from both cycles should be drained"
        )

        # Verify all 4 blocks are on disk
        for h in [100, 200, 300, 400]:
            assert connector._disk_tier.contains(h)

        connector.shutdown()


def test_shutdown_releases_write_locked_blocks() -> None:
    """Verify shutdown() releases all write-locked blocks."""
    with tempfile.TemporaryDirectory(prefix="tiered_sd_wlb_") as disk_dir:
        connector = create_tiered_connector(disk_cache_dir=disk_dir)

        connector.save([0, 1], [100, 200])
        connector.flush()
        connector.sync()

        # Don't manually drain — let shutdown handle it
        connector.shutdown()

        assert len(connector._write_locked_blocks) == 0
        assert len(connector._pending_disk_writes) == 0
