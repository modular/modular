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

import pytest
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.graph import DeviceRef
from max.kv_cache.connectors.tiered_connector import TieredConnector
from max.nn.kv_cache import KVCacheBuffer, KVCacheParams, KVConnectorType


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
        enable_prefix_caching=True,
        kv_connector=KVConnectorType.tiered,
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
        device_buffers=KVCacheBuffer(
            total_num_pages=num_device_blocks, values=device_buffers
        ).all_buffers,
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
    assert connector._host_buffer.pinned, "Host buffer should be pinned memory"
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


# -- Load (CPU hits) --


def test_load_returns_zero_for_empty_cache() -> None:
    connector = create_tiered_connector(page_size=16)
    loaded = connector.load([0, 1, 2], [100, 200, 300])
    assert loaded == 0
    connector.shutdown()


def test_load_finds_cached_blocks() -> None:
    connector = create_tiered_connector(page_size=16)
    connector.save([0, 1, 2], [100, 200, 300])
    connector.flush()

    loaded = connector.load([3, 4, 5], [100, 200, 300])
    assert loaded == 3
    connector.shutdown()


def test_load_stops_at_first_miss() -> None:
    connector = create_tiered_connector(page_size=16)
    connector.save([0], [100])
    connector.save([2], [300])
    connector.flush()

    loaded = connector.load([3, 4, 5], [100, 200, 300])
    assert loaded == 1
    connector.shutdown()


def test_load_full_round_trip() -> None:
    """Verify full prefix cache hit: save -> load round-trip."""
    connector = create_tiered_connector(page_size=16)
    connector.save([0, 1], [100, 200])
    connector.flush()

    loaded = connector.load([10, 11], [100, 200])
    assert loaded == 2
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
    """Verify load promotes blocks from disk to CPU when not in CPU cache."""
    with tempfile.TemporaryDirectory(prefix="tiered_promo_") as disk_dir:
        # Very small CPU cache (4 blocks) to force eviction
        connector = create_tiered_connector(
            num_host_blocks=4,
            disk_cache_dir=disk_dir,
        )

        # Save 4 blocks -> fills CPU cache
        connector.save([0, 1, 2, 3], [100, 200, 300, 400])
        connector.flush()
        connector.sync()
        connector._disk_tier.wait_for_writes()
        connector.sync()  # drain write-locked blocks

        # All 4 should be on disk
        for h in [100, 200, 300, 400]:
            assert connector._disk_tier.contains(h)

        # Save 4 more blocks -> evicts the first 4 from CPU
        connector.save([4, 5, 6, 7], [500, 600, 700, 800])
        connector.flush()
        connector.sync()
        connector._disk_tier.wait_for_writes()
        connector.sync()  # drain write-locked blocks

        # The first 4 should still be on disk (write-through)
        for h in [100, 200, 300, 400]:
            assert connector._disk_tier.contains(h)

        # Load hash 100 -> should find it on disk and promote to CPU
        loaded = connector.load([10], [100])
        assert loaded == 1  # disk hit

        connector.shutdown()


# -- Full round-trip --


def test_full_round_trip() -> None:
    """Save -> write-through -> evict from CPU -> load from disk -> to GPU."""
    with tempfile.TemporaryDirectory(prefix="tiered_rt_") as disk_dir:
        connector = create_tiered_connector(
            num_host_blocks=2,
            disk_cache_dir=disk_dir,
        )

        # Save 2 blocks (fills CPU)
        connector.save([0, 1], [100, 200])
        connector.flush()
        connector.sync()
        connector._disk_tier.wait_for_writes()
        connector.sync()  # drain write-locked blocks

        # Save 2 more -> evicts 100, 200 from CPU
        connector.save([2, 3], [300, 400])
        connector.flush()
        connector.sync()
        connector._disk_tier.wait_for_writes()
        connector.sync()  # drain write-locked blocks

        # 100, 200 should still be on disk
        assert connector._disk_tier.contains(100)
        assert connector._disk_tier.contains(200)

        # Load [100, 200] -> both promoted from disk
        loaded = connector.load([20, 21], [100, 200])
        assert loaded == 2

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

        connector.load([10], [100])

        metrics = connector.metrics
        assert metrics.disk_blocks_read >= 1
        assert metrics.h2d_blocks_copied >= 1

        connector.shutdown()


# -- Prefix chain with disk gap --


def test_load_breaks_chain_at_disk_miss() -> None:
    """Verify load stops at first complete miss (not on CPU or disk)."""
    with tempfile.TemporaryDirectory(prefix="tiered_gap_") as disk_dir:
        connector = create_tiered_connector(disk_cache_dir=disk_dir)

        # Only save hash 100 and 300, NOT 200
        connector.save([0], [100])
        connector.save([2], [300])
        connector.flush()
        connector.sync()
        connector._disk_tier.wait_for_writes()

        # Load [100, 200, 300] -> should stop at 200 (miss)
        loaded = connector.load([10, 11, 12], [100, 200, 300])
        assert loaded == 1  # only hash 100

        connector.shutdown()


# -- Shutdown --


def test_shutdown_clears_pending_state() -> None:
    connector = create_tiered_connector()
    connector.save([0, 1], [100, 200])
    connector.shutdown()

    assert len(connector._pending_saves) == 0
    assert len(connector._pending_disk_writes) == 0


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

        # Second connector: same disk dir -> should load metadata
        c2 = create_tiered_connector(disk_cache_dir=disk_dir)
        assert c2._disk_tier.contains(100)
        assert c2._disk_tier.contains(200)

        # Should be able to promote from disk
        loaded = c2.load([10, 11], [100, 200])
        assert loaded == 2

        c2.shutdown()


# -- Zero-copy disk writes (Change 4) --


def test_write_locked_blocks_lifecycle() -> None:
    """Verify _write_locked_blocks tracks in-flight writes and drains correctly."""
    with tempfile.TemporaryDirectory(prefix="tiered_wlb_") as disk_dir:
        connector = create_tiered_connector(disk_cache_dir=disk_dir)

        # Save and flush -> creates pending_disk_writes
        connector.save([0, 1], [100, 200])
        connector.flush()

        assert len(connector._pending_disk_writes) == 2

        # sync() submits writes to disk and tracks in _write_locked_blocks
        connector.sync()

        # After sync, _pending_disk_writes is cleared
        assert len(connector._pending_disk_writes) == 0

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

        # Save and flush -> D2H copies queued
        connector.save([0, 1], [100, 200])
        connector.flush()

        # At this point, host blocks should be pinned (ref_cnt=1)
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
