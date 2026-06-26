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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheBuffer,
    KVCacheParams,
    KVConnectorType,
    MHAKVCacheParams,
    MultiKVCacheBuffer,
    MultiKVCacheParams,
)
from max.nn.kv_cache.cache_params import KVCacheQuantizationConfig
from max.pipelines.kv_cache.connectors.tiered_connector import TieredConnector
from max.pipelines.kv_cache.kv_connector import to_block_hash_bytes
from max.pipelines.kv_cache.paged_kv_cache.cache_manager import (
    PagedKVCacheManager,
)

# Test-fixture helpers for the bytes-only connector boundary: tests pre-date
# Option A and used int hashes directly; ``_b`` wraps a single int and ``_hs``
# wraps a variadic sequence into the canonical 8-byte signed-BE encoding.
_b = to_block_hash_bytes


def _hs(*ns: int) -> list[bytes]:
    return [_b(n) for n in ns]


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
    kv_params = MHAKVCacheParams(
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

    device_values = [
        Buffer(
            shape=[num_device_blocks, *kv_params.shape_per_block],
            dtype=kv_params.dtype,
            device=device,
        )
    ]
    kv_buffer = KVCacheBuffer(
        values=device_values,
        replicates_kv_across_tp=False,
    )

    return TieredConnector(
        devices=[device],
        replica_kv_memory=[kv_buffer.to_memory()],
        total_num_host_blocks=num_host_blocks,
        disk_cache_dir=disk_cache_dir,
        max_disk_size_gb=max_disk_size_gb,
        synchronous_d2h_copy_mode=True,
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


# -- Offload / sync --


def test_offload_transfers_blocks_to_host() -> None:
    connector = create_tiered_connector()
    connector.offload([0, 1], _hs(100, 200))
    assert connector.num_used_host_blocks == 2
    connector.shutdown()


def test_duplicate_hash_not_saved_twice() -> None:
    connector = create_tiered_connector()
    connector.offload([0], _hs(100))
    connector.offload([1], _hs(100))
    assert connector.num_used_host_blocks == 1
    connector.shutdown()


# -- Load (CPU hits) --


def test_load_returns_zero_for_empty_cache() -> None:
    connector = create_tiered_connector(page_size=16)
    loaded = connector.load([0, 1, 2], _hs(100, 200, 300))
    assert loaded == 0
    connector.shutdown()


def test_load_finds_cached_blocks() -> None:
    connector = create_tiered_connector(page_size=16)
    connector.offload([0, 1, 2], _hs(100, 200, 300))

    loaded = connector.load([3, 4, 5], _hs(100, 200, 300))
    assert loaded == 3
    connector.shutdown()


def test_load_stops_at_first_miss() -> None:
    connector = create_tiered_connector(page_size=16)
    connector.offload([0], _hs(100))
    connector.offload([2], _hs(300))

    loaded = connector.load([3, 4, 5], _hs(100, 200, 300))
    assert loaded == 1
    connector.shutdown()


def test_load_full_round_trip() -> None:
    """Verify full prefix cache hit: save -> load round-trip."""
    connector = create_tiered_connector(page_size=16)
    connector.offload([0, 1], _hs(100, 200))

    loaded = connector.load([10, 11], _hs(100, 200))
    assert loaded == 2
    connector.shutdown()


# -- Write-through to disk --


def test_write_through_to_disk() -> None:
    """Verify blocks written to CPU are also written through to disk."""
    with tempfile.TemporaryDirectory(prefix="tiered_wt_") as disk_dir:
        connector = create_tiered_connector(disk_cache_dir=disk_dir)

        connector.offload([0, 1, 2], _hs(100, 200, 300))
        # offload() records pending disk writes; sync() executes them
        connector.wait_for_offloads()

        # Wait for async disk writes to complete
        connector._disk_tier.wait_for_writes()

        assert connector._disk_tier.contains(_b(100))
        assert connector._disk_tier.contains(_b(200))
        assert connector._disk_tier.contains(_b(300))

        # Verify files exist on disk (sharded into per-hash subdirectories)
        bin_files = list(Path(disk_dir).glob("*/*.bin"))
        assert len(bin_files) == 3

        connector.shutdown()


def test_write_through_skips_already_on_disk() -> None:
    """Verify write-through skips blocks already present on disk."""
    with tempfile.TemporaryDirectory(prefix="tiered_skip_") as disk_dir:
        connector = create_tiered_connector(disk_cache_dir=disk_dir)

        # First offload
        connector.offload([0], _hs(100))
        connector.wait_for_offloads()
        connector._disk_tier.wait_for_writes()

        written_before = connector._disk_blocks_written

        # Offload same hash again (from different device block)
        connector.offload([1], _hs(100))
        connector.wait_for_offloads()
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

        # Offload 4 blocks -> fills CPU cache
        connector.offload([0, 1, 2, 3], _hs(100, 200, 300, 400))
        connector.wait_for_offloads()
        connector._disk_tier.wait_for_writes()
        connector.wait_for_offloads()  # drain write-locked blocks

        # All 4 should be on disk
        for h in _hs(100, 200, 300, 400):
            assert connector._disk_tier.contains(h)

        # Offload 4 more blocks -> evicts the first 4 from CPU
        connector.offload([4, 5, 6, 7], _hs(500, 600, 700, 800))
        connector.wait_for_offloads()
        connector._disk_tier.wait_for_writes()
        connector.wait_for_offloads()  # drain write-locked blocks

        # The first 4 should still be on disk (write-through)
        for h in _hs(100, 200, 300, 400):
            assert connector._disk_tier.contains(h)

        # Load hash 100 -> should find it on disk and promote to CPU
        loaded = connector.load([10], _hs(100))
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

        # Offload 2 blocks (fills CPU)
        connector.offload([0, 1], _hs(100, 200))
        connector.wait_for_offloads()
        connector._disk_tier.wait_for_writes()
        connector.wait_for_offloads()  # drain write-locked blocks

        # Offload 2 more -> evicts 100, 200 from CPU
        connector.offload([2, 3], _hs(300, 400))
        connector.wait_for_offloads()
        connector._disk_tier.wait_for_writes()
        connector.wait_for_offloads()  # drain write-locked blocks

        # 100, 200 should still be on disk
        assert connector._disk_tier.contains(_b(100))
        assert connector._disk_tier.contains(_b(200))

        # Load [100, 200] -> both promoted from disk
        loaded = connector.load([20, 21], _hs(100, 200))
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
        connector.offload([0, 1], _hs(100, 200))
        connector.wait_for_offloads()
        connector._disk_tier.wait_for_writes()
        connector.wait_for_offloads()  # drain write-locked blocks

        metrics = connector.metrics
        assert metrics.d2h_blocks_copied == 2
        assert metrics.disk_blocks_written == 2

        # Evict from CPU, then promote from disk
        connector.offload([2, 3], _hs(300, 400))
        connector.wait_for_offloads()
        connector._disk_tier.wait_for_writes()
        connector.wait_for_offloads()  # drain write-locked blocks

        connector.load([10], _hs(100))

        metrics = connector.metrics
        assert metrics.disk_blocks_read >= 1
        assert metrics.h2d_blocks_copied >= 1

        connector.shutdown()


# -- Prefix chain with disk gap --


def test_load_breaks_chain_at_disk_miss() -> None:
    """Verify load stops at first complete miss (not on CPU or disk)."""
    with tempfile.TemporaryDirectory(prefix="tiered_gap_") as disk_dir:
        connector = create_tiered_connector(disk_cache_dir=disk_dir)

        # Only offload hash 100 and 300, NOT 200
        connector.offload([0], _hs(100))
        connector.offload([2], _hs(300))
        connector.wait_for_offloads()
        connector._disk_tier.wait_for_writes()

        # Load [100, 200, 300] -> should stop at 200 (miss)
        loaded = connector.load([10, 11, 12], _hs(100, 200, 300))
        assert loaded == 1  # only hash 100

        connector.shutdown()


# -- Shutdown --


def test_shutdown_clears_pending_state() -> None:
    connector = create_tiered_connector()
    connector.offload([0, 1], _hs(100, 200))
    connector.shutdown()

    assert len(connector._pending_disk_writes) == 0


# -- Reset --


def test_reset_prefix_cache_clears_cpu_and_disk() -> None:
    with tempfile.TemporaryDirectory(prefix="tiered_reset_") as disk_dir:
        connector = create_tiered_connector(disk_cache_dir=disk_dir)

        connector.offload([0, 1], _hs(100, 200))
        connector.wait_for_offloads()
        connector._disk_tier.wait_for_writes()
        assert connector.num_used_host_blocks == 2
        assert connector._disk_tier.contains(_b(100))

        connector.reset_prefix_cache()

        assert connector.num_used_host_blocks == 0
        assert not connector._disk_tier.contains(_b(100))
        assert not connector._disk_tier.contains(_b(200))

        connector.shutdown()


# -- Warm restart --


def test_warm_restart_loads_disk_cache() -> None:
    """Verify a new TieredConnector finds blocks persisted by a previous one."""
    with tempfile.TemporaryDirectory(prefix="tiered_warm_") as disk_dir:
        # First connector: write blocks to disk
        c1 = create_tiered_connector(disk_cache_dir=disk_dir)
        c1.offload([0, 1], _hs(100, 200))
        c1.wait_for_offloads()
        c1._disk_tier.wait_for_writes()
        c1.shutdown()  # saves metadata

        # Second connector: same disk dir -> should load metadata
        c2 = create_tiered_connector(disk_cache_dir=disk_dir)
        assert c2._disk_tier.contains(_b(100))
        assert c2._disk_tier.contains(_b(200))

        # Should be able to promote from disk
        loaded = c2.load([10, 11], _hs(100, 200))
        assert loaded == 2

        c2.shutdown()


# -- Zero-copy disk writes (Change 4) --


def test_write_locked_blocks_lifecycle() -> None:
    """Verify _write_locked_blocks tracks in-flight writes and drains correctly."""
    with tempfile.TemporaryDirectory(prefix="tiered_wlb_") as disk_dir:
        connector = create_tiered_connector(disk_cache_dir=disk_dir)

        # Offload -> creates pending_disk_writes
        connector.offload([0, 1], _hs(100, 200))

        assert len(connector._pending_disk_writes) == 1
        pending_disk_write = connector._pending_disk_writes[0]
        assert len(pending_disk_write.host_blocks) == 2

        # Ensure that the d2h copies have completed
        for device in connector._devices:
            device.synchronize()

        # The event should be ready
        assert pending_disk_write.d2h_copy_complete_event.is_ready()

        # sync() submits writes to disk and tracks in _write_locked_blocks
        connector.wait_for_offloads()

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

        # Offload -> D2H copies queued
        connector.offload([0, 1], _hs(100, 200))

        # At this point, host blocks should be pinned (ref_cnt=1)
        for pending_disk_write in connector._pending_disk_writes:
            for host_block in pending_disk_write.host_blocks:
                assert host_block.ref_cnt > 0, (
                    "Host block should be pinned for disk write safety"
                )

        # After sync() + wait, blocks should eventually be released
        connector.wait_for_offloads()
        connector._disk_tier.wait_for_writes()
        connector._drain_completed_writes()

        assert len(connector._write_locked_blocks) == 0

        connector.shutdown()


def test_drain_completed_writes_across_sync_cycles() -> None:
    """Verify _drain_completed_writes releases blocks from previous cycles."""
    with tempfile.TemporaryDirectory(prefix="tiered_drain_") as disk_dir:
        connector = create_tiered_connector(disk_cache_dir=disk_dir)

        # Cycle 1: offload + sync
        connector.offload([0, 1], _hs(100, 200))
        connector.wait_for_offloads()

        # Wait for cycle 1 writes to complete
        connector._disk_tier.wait_for_writes()

        # Cycle 2: offload + sync — should drain cycle 1's blocks
        connector.offload([2, 3], _hs(300, 400))
        connector.wait_for_offloads()  # calls _drain_completed_writes() internally

        # Wait for cycle 2
        connector._disk_tier.wait_for_writes()
        connector._drain_completed_writes()

        assert len(connector._write_locked_blocks) == 0, (
            "All blocks from both cycles should be drained"
        )

        # Verify all 4 blocks are on disk
        for h in _hs(100, 200, 300, 400):
            assert connector._disk_tier.contains(h)

        connector.shutdown()


def test_shutdown_releases_write_locked_blocks() -> None:
    """Verify shutdown() releases all write-locked blocks."""
    with tempfile.TemporaryDirectory(prefix="tiered_sd_wlb_") as disk_dir:
        connector = create_tiered_connector(disk_cache_dir=disk_dir)

        connector.offload([0, 1], _hs(100, 200))
        connector.wait_for_offloads()

        # Don't manually drain — let shutdown handle it
        connector.shutdown()

        assert len(connector._write_locked_blocks) == 0
        assert len(connector._pending_disk_writes) == 0


# -- Bit-exact data round-trip (SERVOPT-1420) --
#
# The tests above verify block *accounting* (loaded counts, contains(),
# ref-count lifecycle) but never assert that the *bytes* of a cached block
# survive offload -> D2H -> disk -> evict -> disk-read -> H2D unchanged.  That
# gap let an accuracy collapse through that only appears with the tiered
# connector and an FP8 KV cache.  For FP8 the connector receives TWO device
# buffers (values + scales) which the BlockOffloadEngine concatenates into one
# packed host page; bf16 receives only one buffer and never exercises that
# packed multi-buffer path.  These tests pin the byte-exactness of that path
# for both layouts.


def _bytes_per_block(buf: Buffer) -> int:
    return buf.num_elements * buf.dtype.size_in_bytes // buf.shape[0]


def _write_block_pattern(buf: Buffer, block_id: int, seed: int) -> np.ndarray:
    """Write a deterministic uint8 pattern into one device block.

    Returns the pattern (ground truth) as a 1-D uint8 array.
    """
    nbytes = _bytes_per_block(buf)
    pattern = np.random.RandomState(seed).randint(
        0, 256, size=(nbytes,), dtype=np.uint8
    )
    host = Buffer.from_numpy(pattern.copy())
    buf.view(dtype=DType.uint8, shape=[buf.shape[0], nbytes])[
        block_id, :
    ].inplace_copy_from(host.to(buf.device))
    return pattern


def _read_block_bytes(buf: Buffer, block_id: int) -> np.ndarray:
    nbytes = _bytes_per_block(buf)
    return (
        buf.view(dtype=DType.uint8, shape=[buf.shape[0], nbytes])[block_id, :]
        .to_numpy()
        .reshape(-1)
        .copy()
    )


def _make_connector_and_buffers(
    *,
    use_fp8: bool,
    disk_cache_dir: str,
    num_device_blocks: int = 32,
    num_host_blocks: int = 4,
) -> tuple[TieredConnector, list[Buffer]]:
    """Build a TieredConnector wired exactly like production.

    bf16: device_buffers = [values].
    fp8 : device_buffers = [values, scales], mirroring the gemma-4-31B
    sliding-cache layout (scale_dtype=float32, granularity=64, head_dim=256).
    """
    device = Accelerator()
    page_size = 16
    num_layers = 2

    if use_fp8:
        dtype = DType.float8_e4m3fn
        head_dim = 256
        n_kv_heads = 4
        quant_cfg: KVCacheQuantizationConfig | None = KVCacheQuantizationConfig(
            scale_dtype=DType.float32, quantization_granularity=64
        )
    else:
        dtype = DType.bfloat16
        head_dim = 128
        n_kv_heads = 8
        quant_cfg = None

    kv_params = MHAKVCacheParams(
        dtype=dtype,
        num_layers=num_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        enable_prefix_caching=True,
        kv_connector=KVConnectorType.tiered,
        host_kvcache_swap_space_gb=999,
        page_size=page_size,
        devices=[DeviceRef.GPU()],
        kvcache_quant_config=quant_cfg,
    )

    values = [
        Buffer(
            shape=[num_device_blocks, *kv_params.shape_per_block],
            dtype=kv_params.dtype,
            device=device,
        )
    ]
    scales = None
    if use_fp8:
        assert quant_cfg is not None
        scales = [
            Buffer(
                shape=[num_device_blocks, *kv_params.shape_per_scale_block],
                dtype=quant_cfg.scale_dtype,
                device=device,
            )
        ]
    kv_buffer = KVCacheBuffer(
        values=values,
        scales=scales,
        replicates_kv_across_tp=False,
    )

    connector = TieredConnector(
        devices=[device],
        replica_kv_memory=[kv_buffer.to_memory()],
        total_num_host_blocks=num_host_blocks,
        disk_cache_dir=disk_cache_dir,
        max_disk_size_gb=1.0,
        synchronous_d2h_copy_mode=True,
    )
    return connector, kv_buffer.all_buffers


def _evict_cpu_prefix(
    connector: TieredConnector,
    device_buffers: list[Buffer],
    scratch_block: int,
    num_host_blocks: int,
) -> None:
    """Fill every host slot with dummy blocks so the prior prefix is evicted
    from the CPU tier (and only reachable via disk)."""
    for k in range(num_host_blocks):
        for b in device_buffers:
            _write_block_pattern(b, scratch_block, seed=987654)
        connector.offload([scratch_block], _hs(900_000 + k))
        connector.wait_for_offloads()
        connector._disk_tier.wait_for_writes()
        connector.wait_for_offloads()


@pytest.mark.parametrize(
    "use_fp8",
    [
        pytest.param(False, id="bf16"),
        pytest.param(True, id="fp8_values_and_scales"),
    ],
)
def test_disk_round_trip_is_bit_exact(use_fp8: bool) -> None:
    """offload -> disk -> evict-from-CPU -> load must return identical bytes.

    Regression guard for SERVOPT-1420: the tiered connector's packed
    values+scales disk round-trip must be byte-for-byte exact.  Asserts the
    full block (values region AND, for fp8, the scales region) is preserved.
    """
    num_host_blocks = 4
    with tempfile.TemporaryDirectory(prefix="tiered_bitexact_") as disk_dir:
        connector, device_buffers = _make_connector_and_buffers(
            use_fp8=use_fp8,
            disk_cache_dir=disk_dir,
            num_host_blocks=num_host_blocks,
        )

        hashes = _hs(1001, 1002, 1003)
        src_blocks = [0, 1, 2]
        ground_truth: dict[bytes, np.ndarray] = {}
        for i, (bid, h) in enumerate(zip(src_blocks, hashes, strict=False)):
            parts = [
                _write_block_pattern(b, bid, seed=1000 * (i + 1) + j)
                for j, b in enumerate(device_buffers)
            ]
            ground_truth[h] = np.concatenate(parts)

        # Offload -> push through to disk.
        connector.offload(src_blocks, hashes)
        connector.wait_for_offloads()
        connector._disk_tier.wait_for_writes()
        connector.wait_for_offloads()
        for h in hashes:
            assert connector._disk_tier.contains(h)

        # Evict the CPU tier so the next load MUST come from disk.
        _evict_cpu_prefix(connector, device_buffers, 10, num_host_blocks)
        for h in hashes:
            assert h not in connector._host_block_pool.prefix_cache, (
                f"hash {h.hex()} should be evicted from CPU"
            )
            assert connector._disk_tier.contains(h)

        # Zero the destination device blocks so a no-op load is detectable.
        dst_blocks = [20, 21, 22]
        for bid in dst_blocks:
            for b in device_buffers:
                _write_block_pattern(b, bid, seed=0)

        loaded = connector.load(dst_blocks, hashes)
        assert loaded == len(hashes), (
            f"expected all {len(hashes)} blocks loaded from disk, got {loaded}"
        )

        for dst_bid, h in zip(dst_blocks, hashes, strict=False):
            actual = np.concatenate(
                [_read_block_bytes(b, dst_bid) for b in device_buffers]
            )
            np.testing.assert_array_equal(
                actual,
                ground_truth[h],
                err_msg=(
                    f"block hash {h.hex()} corrupted on disk round-trip "
                    f"(use_fp8={use_fp8})"
                ),
            )

        connector.shutdown()


@pytest.mark.parametrize(
    "use_fp8",
    [
        pytest.param(False, id="bf16"),
        pytest.param(True, id="fp8_values_and_scales"),
    ],
)
def test_mixed_cpu_and_disk_chain_is_bit_exact(use_fp8: bool) -> None:
    """A prefix chain that is part CPU-resident and part disk-only must load
    bit-exact across both tiers in a single load() call.

    Exercises the load() ordering where CPU hits and freshly-promoted disk
    hits coexist (the path that frees all hit blocks before issuing H2D).
    """
    num_host_blocks = 4
    with tempfile.TemporaryDirectory(prefix="tiered_mixed_") as disk_dir:
        connector, device_buffers = _make_connector_and_buffers(
            use_fp8=use_fp8,
            disk_cache_dir=disk_dir,
            num_host_blocks=num_host_blocks,
        )

        hashes = _hs(3000, 3001, 3002, 3003)
        ground_truth: dict[bytes, np.ndarray] = {}
        for i, (bid, h) in enumerate(zip([0, 1, 2, 3], hashes, strict=False)):
            parts = [
                _write_block_pattern(b, bid, seed=7000 * (i + 1) + j)
                for j, b in enumerate(device_buffers)
            ]
            ground_truth[h] = np.concatenate(parts)

        def restore(bid: int, h: bytes) -> None:
            off = 0
            for b in device_buffers:
                n = _bytes_per_block(b)
                seg = ground_truth[h][off : off + n]
                host = Buffer.from_numpy(seg.copy())
                b.view(dtype=DType.uint8, shape=[b.shape[0], n])[
                    bid, :
                ].inplace_copy_from(host.to(b.device))
                off += n

        # Push h2,h3 to disk, evict everything, then make h0,h1 CPU-resident.
        connector.offload([2, 3], _hs(3002, 3003))
        connector.wait_for_offloads()
        connector._disk_tier.wait_for_writes()
        connector.wait_for_offloads()
        _evict_cpu_prefix(connector, device_buffers, 12, num_host_blocks)
        restore(0, _b(3000))
        restore(1, _b(3001))
        connector.offload([0, 1], _hs(3000, 3001))
        connector.wait_for_offloads()
        connector._disk_tier.wait_for_writes()
        connector.wait_for_offloads()

        in_cpu = [
            h for h in hashes if h in connector._host_block_pool.prefix_cache
        ]
        assert in_cpu == _hs(3000, 3001), (
            f"expected 3000,3001 CPU-resident, got {in_cpu}"
        )
        for h in hashes:
            assert connector._disk_tier.contains(h)

        dst_blocks = [20, 21, 22, 23]
        for bid in dst_blocks:
            for b in device_buffers:
                _write_block_pattern(b, bid, seed=0)

        loaded = connector.load(dst_blocks, hashes)
        assert loaded == len(hashes)

        for dst_bid, h in zip(dst_blocks, hashes, strict=False):
            actual = np.concatenate(
                [_read_block_bytes(b, dst_bid) for b in device_buffers]
            )
            np.testing.assert_array_equal(
                actual,
                ground_truth[h],
                err_msg=(
                    f"block hash {h.hex()} corrupted on mixed CPU+disk load "
                    f"(use_fp8={use_fp8})"
                ),
            )

        connector.shutdown()


# -- Multi-cache offload: every cache must be offloaded/restored (SERVOPT-1420) --
#
# gemma4 uses MultiKVCacheParams (sliding idx0 + global idx1) on a SINGLE block
# pool with a SINGLE lookup table, so one physical block id indexes into BOTH
# caches.  The connector moves blocks by physical id; if it offloads only the
# primary cache (the old `replica_device_buffers[0].all_buffers` shortcut,
# TODO SERVOPT-1254), a prefix-cache hit served from host/disk reloads only the
# primary half of a reused block and leaves the other caches' halves STALE
# (whatever the block's previous occupant wrote) -> corrupts attention for the
# layers backed by those caches.  These tests pin the fix: the connector must
# offload/restore EVERY cache's buffers.


@dataclass(frozen=True)
class _DiskConnectorConfig:
    """Hashable stand-in for the tiered connector config."""

    disk_offload_dir: str
    disk_offload_max_gb: float
    disk_offload_direct_io: bool
    host_kvcache_swap_space_gb: float


def _fp8_cache_params(
    cfg: _DiskConnectorConfig,
    *,
    n_kv_heads: int,
    head_dim: int,
    num_layers: int = 2,
) -> KVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.float8_e4m3fn,
        num_layers=num_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        enable_prefix_caching=True,
        kv_connector=KVConnectorType.tiered,
        kv_connector_config=cfg,
        host_kvcache_swap_space_gb=999,
        page_size=16,
        devices=[DeviceRef.GPU()],
        kvcache_quant_config=KVCacheQuantizationConfig(
            scale_dtype=DType.float32, quantization_granularity=64
        ),
    )


def _build_multi_cache_manager(
    disk_dir: str,
    *,
    total_num_pages: int = 32,
    total_num_host_pages: int = 4,
) -> PagedKVCacheManager:
    """Two fp8 caches (gemma4-like sliding idx0 + global idx1) on one pool with
    a tiered connector."""
    cfg = _DiskConnectorConfig(
        disk_offload_dir=disk_dir,
        disk_offload_max_gb=1.0,
        disk_offload_direct_io=False,
        host_kvcache_swap_space_gb=999.0,
    )
    sliding = _fp8_cache_params(cfg, n_kv_heads=4, head_dim=256)  # idx0
    global_ = _fp8_cache_params(cfg, n_kv_heads=2, head_dim=512)  # idx1
    multi = MultiKVCacheParams.from_params(
        {"sliding_attention": sliding, "full_attention": global_}
    )
    session = InferenceSession(devices=[Accelerator()])
    return PagedKVCacheManager(
        params=multi,
        session=session,
        total_num_pages=total_num_pages,
        total_num_host_pages=total_num_host_pages,
        max_batch_size=4,
    )


def test_connector_offloads_every_cache_not_just_primary() -> None:
    """Structural guard: the connector's offload engine must cover ALL caches'
    buffers, not only the primary (sliding) cache.

    Pre-fix (idx0 only) the engine would hold just the sliding cache's
    values+scales (2 buffers); the fix offloads both caches (4 buffers) and
    sizes the host page to the sum across all caches.
    """
    if accelerator_count() == 0:
        pytest.skip("No GPU available")

    with tempfile.TemporaryDirectory(prefix="multi_struct_") as disk_dir:
        mgr = _build_multi_cache_manager(disk_dir)
        connector = mgr._replica[0].connector
        assert isinstance(connector, TieredConnector)
        engine = connector._block_copy_engine
        multi_buffer = mgr.get_device_buffer(0)
        assert isinstance(multi_buffer, MultiKVCacheBuffer)
        cache_buffers = list(multi_buffer.children.values())

        expected_num_buffers = sum(len(kc.all_buffers) for kc in cache_buffers)
        expected_page_bytes = sum(
            _bytes_per_block(b) for kc in cache_buffers for b in kc.all_buffers
        )

        assert len(cache_buffers) == 2, "expected sliding + global caches"
        device_buffers = engine._replicas[0].device_buffers
        assert len(device_buffers) == expected_num_buffers, (
            "connector offload engine must cover every cache's buffers "
            f"(sliding+global); got {len(device_buffers)}, expected "
            f"{expected_num_buffers} -- only the primary cache is offloaded "
            "(SERVOPT-1420 regression)"
        )
        assert engine.host_buffer.shape[1] == expected_page_bytes, (
            "host page must span all caches' packed bytes; got "
            f"{engine.host_buffer.shape[1]}, expected {expected_page_bytes}"
        )
        connector.shutdown()


def test_multi_cache_disk_round_trip_restores_all_caches() -> None:
    """Behavioral guard: a shared physical block reloaded from disk must restore
    EVERY cache's half bit-exact, not just the primary's.

    Drives the real manager-constructed connector: write distinct data into all
    caches' buffers at one physical block id, offload -> evict host -> overwrite
    the device block with a distinct reuse pattern (simulating block reuse) ->
    load.  Pre-fix the global cache's half is left as the reuse pattern (stale).
    """
    if accelerator_count() == 0:
        pytest.skip("No GPU available")

    with tempfile.TemporaryDirectory(prefix="multi_rt_") as disk_dir:
        mgr = _build_multi_cache_manager(disk_dir, total_num_host_pages=4)
        connector = mgr._replica[0].connector
        assert isinstance(connector, TieredConnector)

        # Flatten all caches' buffers in engine-packing order: [v0,s0,v1,s1].
        multi_buffer = mgr.get_device_buffer(0)
        assert isinstance(multi_buffer, MultiKVCacheBuffer)
        all_bufs = [
            b for kc in multi_buffer.children.values() for b in kc.all_buffers
        ]
        assert len(all_bufs) == 4

        bid = 0
        block_hash = _b(4242)
        ground_truth = {
            i: _write_block_pattern(b, bid, seed=100 + i * 7)
            for i, b in enumerate(all_bufs)
        }

        connector.offload([bid], [block_hash])
        connector.wait_for_offloads()
        connector._disk_tier.wait_for_writes()
        connector.wait_for_offloads()

        # Evict host so the reload comes from disk.
        for k in range(4):
            for b in all_bufs:
                _write_block_pattern(b, 1, seed=55)
            connector.offload([1], _hs(70_000 + k))
            connector.wait_for_offloads()
            connector._disk_tier.wait_for_writes()
            connector.wait_for_offloads()
        assert connector._disk_tier.contains(block_hash)
        assert block_hash not in connector._host_block_pool.prefix_cache

        # Simulate the physical block being reused: overwrite ALL caches' block
        # `bid` with a distinct nonzero pattern (NOT zeros -- RandomState(0) is
        # nonzero, which would mask a stale-vs-restored check).
        reuse = {
            i: _write_block_pattern(b, bid, seed=314_159 + i)
            for i, b in enumerate(all_bufs)
        }

        loaded = connector.load([bid], [block_hash])
        assert loaded == 1

        for i, b in enumerate(all_bufs):
            got = _read_block_bytes(b, bid)
            cache = "sliding" if i < 2 else "global"
            region = "values" if i % 2 == 0 else "scales"
            assert not np.array_equal(got, reuse[i]) or np.array_equal(
                got, ground_truth[i]
            ), (
                f"{cache}/{region} (buf #{i}) left STALE after disk reload -- "
                "connector did not restore this cache (SERVOPT-1420)"
            )
            np.testing.assert_array_equal(
                got,
                ground_truth[i],
                err_msg=(
                    f"{cache}/{region} (buf #{i}) not restored bit-exact from "
                    "disk on a shared physical block (SERVOPT-1420)"
                ),
            )

        connector.shutdown()
