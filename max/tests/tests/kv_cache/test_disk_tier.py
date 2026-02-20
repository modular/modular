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

"""Unit tests for DiskTier"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from max.kv_cache.connectors.disk_tier import DiskTier


@pytest.fixture()
def cache_dir(tmp_path: Path) -> str:
    return str(tmp_path / "disk_cache")


def _make_block(
    num_devices: int, block_shape: tuple[int, ...], seed: int = 0
) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    return [
        rng.randint(0, 256, size=block_shape, dtype=np.uint8)
        for _ in range(num_devices)
    ]


# -- Write/read round-trip correctness --


def test_read_missing_hash_raises(cache_dir: str) -> None:
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        num_devices=1,
        max_disk_size_bytes=10_000,
    )
    dest = [np.zeros((16,), dtype=np.uint8)]
    with pytest.raises(KeyError):
        tier.read_block_async(block_hash=999, dest=dest)

    tier.shutdown()


def test_write_read_roundtrip(cache_dir: str) -> None:
    block_shape = (64,)
    num_devices = 2
    block_nbytes = 64  # uint8 so nbytes == nelements
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=block_nbytes,
        num_devices=num_devices,
        max_disk_size_bytes=10 * block_nbytes * num_devices,
    )

    src = _make_block(num_devices, block_shape, seed=42)
    tier.write_block_async(block_hash=100, src=src)
    tier.wait_for_writes()

    dest = [np.zeros(block_shape, dtype=np.uint8) for _ in range(num_devices)]
    future = tier.read_block_async(block_hash=100, dest=dest)
    future.result()

    for s, d in zip(src, dest, strict=False):
        np.testing.assert_array_equal(s, d)

    tier.shutdown()


def test_write_read_multiple_devices(cache_dir: str) -> None:
    block_shape = (32,)
    num_devices = 4
    block_nbytes = 32
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=block_nbytes,
        num_devices=num_devices,
        max_disk_size_bytes=10 * block_nbytes * num_devices,
    )

    src = _make_block(num_devices, block_shape, seed=7)
    tier.write_block_async(block_hash=1, src=src)
    tier.wait_for_writes()

    dest = [np.zeros(block_shape, dtype=np.uint8) for _ in range(num_devices)]
    future = tier.read_block_async(block_hash=1, dest=dest)
    future.result()

    for device_idx, (s, d) in enumerate(zip(src, dest, strict=False)):
        np.testing.assert_array_equal(
            s, d, err_msg=f"Mismatch on device {device_idx}"
        )

    tier.shutdown()


def test_write_read_with_scales(cache_dir: str) -> None:
    block_shape = (64,)
    scale_shape = (8,)
    num_devices = 1
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=64,
        num_devices=num_devices,
        max_disk_size_bytes=10_000,
        has_scales=True,
        scale_block_nbytes=8,
    )

    src = _make_block(num_devices, block_shape, seed=1)
    scale_src = _make_block(num_devices, scale_shape, seed=2)
    tier.write_block_async(block_hash=200, src=src, scale_src=scale_src)
    tier.wait_for_writes()

    dest = [np.zeros(block_shape, dtype=np.uint8)]
    scale_dest = [np.zeros(scale_shape, dtype=np.uint8)]
    future = tier.read_block_async(
        block_hash=200, dest=dest, scale_dest=scale_dest
    )
    future.result()

    np.testing.assert_array_equal(src[0], dest[0])
    np.testing.assert_array_equal(scale_src[0], scale_dest[0])

    tier.shutdown()


# -- contains() behavior --


def test_contains_after_write(cache_dir: str) -> None:
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        num_devices=1,
        max_disk_size_bytes=10_000,
    )
    assert not tier.contains(42)

    src = _make_block(1, (16,), seed=0)
    tier.write_block_async(block_hash=42, src=src)

    # Should be True immediately (pending hash)
    assert tier.contains(42)
    tier.wait_for_writes()
    # Still True after write completes
    assert tier.contains(42)

    tier.shutdown()


def test_contains_after_eviction(cache_dir: str) -> None:
    # Only room for 1 block
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        num_devices=1,
        max_disk_size_bytes=16,
    )
    src1 = _make_block(1, (16,), seed=1)
    tier.write_block_async(block_hash=1, src=src1)
    tier.wait_for_writes()
    assert tier.contains(1)

    # Writing a second block should evict the first
    src2 = _make_block(1, (16,), seed=2)
    tier.write_block_async(block_hash=2, src=src2)
    tier.wait_for_writes()
    assert tier.contains(2)
    assert not tier.contains(1)

    tier.shutdown()


# -- LRU eviction --


def test_lru_eviction_order(cache_dir: str) -> None:
    # Room for exactly 2 blocks
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        num_devices=1,
        max_disk_size_bytes=32,
    )

    for h in [10, 20]:
        src = _make_block(1, (16,), seed=h)
        tier.write_block_async(block_hash=h, src=src)
        tier.wait_for_writes()  # serialize to guarantee LRU order

    assert tier.contains(10)
    assert tier.contains(20)

    # Write a third — should evict hash 10 (LRU)
    src = _make_block(1, (16,), seed=30)
    tier.write_block_async(block_hash=30, src=src)
    tier.wait_for_writes()

    assert not tier.contains(10)
    assert tier.contains(20)
    assert tier.contains(30)

    tier.shutdown()


# -- reset() clears all blocks --


def test_reset_clears_all(cache_dir: str) -> None:
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        num_devices=1,
        max_disk_size_bytes=10_000,
    )
    for h in range(5):
        src = _make_block(1, (16,), seed=h)
        tier.write_block_async(block_hash=h, src=src)
    tier.wait_for_writes()

    tier.reset()

    for h in range(5):
        assert not tier.contains(h)

    tier.shutdown()


# -- Metadata persistence across DiskTier instances --


def test_persistence_reload(cache_dir: str) -> None:
    tier1 = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        num_devices=1,
        max_disk_size_bytes=10_000,
    )
    src = _make_block(1, (16,), seed=99)
    tier1.write_block_async(block_hash=99, src=src)
    tier1.shutdown()  # saves metadata

    # Create a new DiskTier pointing to same dir
    tier2 = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        num_devices=1,
        max_disk_size_bytes=10_000,
    )
    assert tier2.contains(99)

    # Verify data is readable
    dest = [np.zeros((16,), dtype=np.uint8)]
    future = tier2.read_block_async(block_hash=99, dest=dest)
    future.result()
    np.testing.assert_array_equal(src[0], dest[0])

    tier2.shutdown()


def test_persistence_config_mismatch(cache_dir: str) -> None:
    tier1 = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        num_devices=1,
        max_disk_size_bytes=10_000,
    )
    src = _make_block(1, (16,), seed=7)
    tier1.write_block_async(block_hash=7, src=src)
    tier1.shutdown()

    # Different block_nbytes → cache should be wiped
    tier2 = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=32,
        num_devices=1,
        max_disk_size_bytes=10_000,
    )
    assert not tier2.contains(7)
    tier2.shutdown()


# -- Duplicate write detection --


def test_no_duplicate_write(cache_dir: str) -> None:
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        num_devices=1,
        max_disk_size_bytes=10_000,
    )
    src = _make_block(1, (16,), seed=5)
    tier.write_block_async(block_hash=5, src=src)
    tier.wait_for_writes()

    # Second write with same hash should be a no-op
    src2 = _make_block(1, (16,), seed=6)
    tier.write_block_async(block_hash=5, src=src2)
    tier.wait_for_writes()

    # Read should return original data
    dest = [np.zeros((16,), dtype=np.uint8)]
    future = tier.read_block_async(block_hash=5, dest=dest)
    future.result()
    np.testing.assert_array_equal(src[0], dest[0])

    tier.shutdown()


# -- Block removal --


def test_remove_block(cache_dir: str) -> None:
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        num_devices=1,
        max_disk_size_bytes=10_000,
    )
    src = _make_block(1, (16,), seed=11)
    tier.write_block_async(block_hash=11, src=src)
    tier.wait_for_writes()
    assert tier.contains(11)

    tier.remove(11)
    assert not tier.contains(11)

    tier.shutdown()


# -- write_block_async() returns Future | None --


def test_returns_future_for_new_block(cache_dir: str) -> None:
    """write_block_async returns a Future when a write is submitted."""
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        num_devices=1,
        max_disk_size_bytes=10_000,
    )
    src = _make_block(1, (16,), seed=1)
    result = tier.write_block_async(block_hash=1, src=src)
    assert result is not None, "Should return a Future for new block"
    result.result()  # wait for completion
    tier.shutdown()


def test_returns_none_for_duplicate_block(cache_dir: str) -> None:
    """write_block_async returns None when the block is already on disk."""
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        num_devices=1,
        max_disk_size_bytes=10_000,
    )
    src = _make_block(1, (16,), seed=1)
    tier.write_block_async(block_hash=1, src=src)
    tier.wait_for_writes()

    # Second write with same hash → None (already on disk)
    src2 = _make_block(1, (16,), seed=2)
    result = tier.write_block_async(block_hash=1, src=src2)
    assert result is None, "Should return None for duplicate block"
    tier.shutdown()


def test_returns_none_for_pending_block(cache_dir: str) -> None:
    """write_block_async returns None when a write is already in-flight."""
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        num_devices=1,
        max_disk_size_bytes=10_000,
    )
    src1 = _make_block(1, (16,), seed=1)
    first = tier.write_block_async(block_hash=1, src=src1)
    assert first is not None

    # While first write is in-flight, submit same hash
    src2 = _make_block(1, (16,), seed=2)
    second = tier.write_block_async(block_hash=1, src=src2)
    assert second is None, "Should return None for in-flight block"

    tier.wait_for_writes()
    tier.shutdown()


# -- O_DIRECT disk I/O support --


def test_direct_io_fallback_when_unavailable(cache_dir: str) -> None:
    """DiskTier gracefully disables O_DIRECT when not available."""
    import os

    has_odirect = hasattr(os, "O_DIRECT")

    # Even if we request direct_io, it should work (either natively
    # or by falling back to buffered I/O).
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        num_devices=1,
        max_disk_size_bytes=10_000,
        use_direct_io=True,
    )

    if not has_odirect:
        assert not tier._use_direct_io, "Should fall back to buffered I/O"

    # Regardless of mode, read/write should still work
    src = _make_block(1, (16,), seed=42)
    tier.write_block_async(block_hash=100, src=src)
    tier.wait_for_writes()

    dest = [np.zeros((16,), dtype=np.uint8)]
    future = tier.read_block_async(block_hash=100, dest=dest)
    future.result()
    np.testing.assert_array_equal(src[0], dest[0])

    tier.shutdown()


@pytest.mark.skipif(
    not hasattr(__import__("os"), "O_DIRECT"),
    reason="O_DIRECT not available on this platform",
)
def test_direct_io_roundtrip(cache_dir: str) -> None:
    """Verify O_DIRECT read/write roundtrip when available.

    Uses 4096-byte blocks (FS-aligned) and relies on numpy's aligned
    allocator so the buffer itself meets O_DIRECT requirements.
    """
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=4096,
        num_devices=1,
        max_disk_size_bytes=10 * 4096,
        use_direct_io=True,
    )

    if not tier._use_direct_io:
        pytest.skip("O_DIRECT disabled due to alignment constraints")

    src = _make_block(1, (4096,), seed=99)
    tier.write_block_async(block_hash=50, src=src)
    tier.wait_for_writes()

    dest = [np.zeros((4096,), dtype=np.uint8)]
    future = tier.read_block_async(block_hash=50, dest=dest)
    future.result()
    np.testing.assert_array_equal(src[0], dest[0])

    tier.shutdown()


def test_direct_io_disabled_on_unaligned_blocks(cache_dir: str) -> None:
    """Verify O_DIRECT is disabled when block size is not FS-aligned."""
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=17,  # Not aligned to any FS block size
        num_devices=1,
        max_disk_size_bytes=10_000,
        use_direct_io=True,
    )
    # 17 bytes is not aligned to 4096 (typical FS block size), so
    # O_DIRECT should be disabled automatically.
    assert not tier._use_direct_io, (
        "Should disable O_DIRECT for unaligned block sizes"
    )
    tier.shutdown()
