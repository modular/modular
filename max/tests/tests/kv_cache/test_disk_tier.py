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
import numpy.typing as npt
import pytest
from max.pipelines.kv_cache.connectors.disk_tier import DiskTier
from max.pipelines.kv_cache.kv_connector import to_block_hash_bytes


@pytest.fixture()
def cache_dir(tmp_path: Path) -> str:
    return str(tmp_path / "disk_cache")


def _make_block(block_nbytes: int, seed: int = 0) -> npt.NDArray[np.uint8]:
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(block_nbytes,), dtype=np.uint8)


# -- Write/read round-trip correctness --


def test_read_missing_hash_raises(cache_dir: str) -> None:
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        max_disk_size_bytes=10_000,
    )
    dest = np.zeros((16,), dtype=np.uint8)
    with pytest.raises(KeyError):
        tier.read_block_async(block_hash=to_block_hash_bytes(999), dest=dest)

    tier.shutdown()


def test_write_read_roundtrip(cache_dir: str) -> None:
    block_nbytes = 64
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=block_nbytes,
        max_disk_size_bytes=10 * block_nbytes,
    )

    src = _make_block(block_nbytes, seed=42)
    tier.write_block_async(block_hash=to_block_hash_bytes(100), src=src)
    tier.wait_for_writes()

    dest = np.zeros(block_nbytes, dtype=np.uint8)
    future = tier.read_block_async(
        block_hash=to_block_hash_bytes(100), dest=dest
    )
    future.result()

    np.testing.assert_array_equal(src, dest)

    tier.shutdown()


def test_write_read_large_block(cache_dir: str) -> None:
    block_nbytes = 1024
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=block_nbytes,
        max_disk_size_bytes=10 * block_nbytes,
    )

    src = _make_block(block_nbytes, seed=7)
    tier.write_block_async(block_hash=to_block_hash_bytes(1), src=src)
    tier.wait_for_writes()

    dest = np.zeros(block_nbytes, dtype=np.uint8)
    future = tier.read_block_async(block_hash=to_block_hash_bytes(1), dest=dest)
    future.result()

    np.testing.assert_array_equal(src, dest)

    tier.shutdown()


# -- contains() behavior --


def test_contains_after_write(cache_dir: str) -> None:
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        max_disk_size_bytes=10_000,
    )
    assert not tier.contains(to_block_hash_bytes(42))

    src = _make_block(16, seed=0)
    tier.write_block_async(block_hash=to_block_hash_bytes(42), src=src)

    # Pending writes are not visible via contains()
    assert not tier.contains(to_block_hash_bytes(42))
    tier.wait_for_writes()
    # True after write completes
    assert tier.contains(to_block_hash_bytes(42))

    tier.shutdown()


def test_contains_after_eviction(cache_dir: str) -> None:
    # Only room for 1 block
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        max_disk_size_bytes=16,
    )
    src1 = _make_block(16, seed=1)
    tier.write_block_async(block_hash=to_block_hash_bytes(1), src=src1)
    tier.wait_for_writes()
    assert tier.contains(to_block_hash_bytes(1))

    # Writing a second block should evict the first
    src2 = _make_block(16, seed=2)
    tier.write_block_async(block_hash=to_block_hash_bytes(2), src=src2)
    tier.wait_for_writes()
    assert tier.contains(to_block_hash_bytes(2))
    assert not tier.contains(to_block_hash_bytes(1))

    tier.shutdown()


# -- LRU eviction --


def test_lru_eviction_order(cache_dir: str) -> None:
    # Room for exactly 2 blocks
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        max_disk_size_bytes=32,
    )

    for h in [10, 20]:
        src = _make_block(16, seed=h)
        tier.write_block_async(block_hash=to_block_hash_bytes(h), src=src)
        tier.wait_for_writes()

    assert tier.contains(to_block_hash_bytes(10))
    assert tier.contains(to_block_hash_bytes(20))

    # Write a third — should evict hash 10 (LRU)
    src = _make_block(16, seed=30)
    tier.write_block_async(block_hash=to_block_hash_bytes(30), src=src)
    tier.wait_for_writes()

    assert not tier.contains(to_block_hash_bytes(10))
    assert tier.contains(to_block_hash_bytes(20))
    assert tier.contains(to_block_hash_bytes(30))

    tier.shutdown()


# -- reset() clears all blocks --


def test_reset_clears_all(cache_dir: str) -> None:
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        max_disk_size_bytes=10_000,
    )
    for h in range(5):
        src = _make_block(16, seed=h)
        tier.write_block_async(block_hash=to_block_hash_bytes(h), src=src)
    tier.wait_for_writes()

    tier.reset()

    for h in range(5):
        assert not tier.contains(to_block_hash_bytes(h))

    tier.shutdown()


# -- Metadata persistence across DiskTier instances --


def test_persistence_reload(cache_dir: str) -> None:
    tier1 = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        max_disk_size_bytes=10_000,
    )
    src = _make_block(16, seed=99)
    tier1.write_block_async(block_hash=to_block_hash_bytes(99), src=src)
    tier1.shutdown()

    # Create a new DiskTier pointing to same dir
    tier2 = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        max_disk_size_bytes=10_000,
    )
    assert tier2.contains(to_block_hash_bytes(99))

    # Verify data is readable
    dest = np.zeros(16, dtype=np.uint8)
    future = tier2.read_block_async(
        block_hash=to_block_hash_bytes(99), dest=dest
    )
    future.result()
    np.testing.assert_array_equal(src, dest)

    tier2.shutdown()


def test_block_size_change_is_not_detected(cache_dir: str) -> None:
    """Reusing a cache_dir across a block_nbytes change is unsupported.

    No metadata is persisted, so a block-size change is not detected and the
    stale blocks remain indexed. Callers must point a changed configuration at
    a fresh cache_dir.
    """
    tier1 = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        max_disk_size_bytes=10_000,
    )
    tier1.write_block_async(
        block_hash=to_block_hash_bytes(7), src=_make_block(16, seed=7)
    )
    tier1.wait_for_writes()
    tier1.shutdown()

    tier2 = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=32,
        max_disk_size_bytes=10_000,
    )
    assert tier2.contains(to_block_hash_bytes(7))
    tier2.shutdown()


# -- Duplicate write detection --


def test_no_duplicate_write(cache_dir: str) -> None:
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        max_disk_size_bytes=10_000,
    )
    src = _make_block(16, seed=5)
    tier.write_block_async(block_hash=to_block_hash_bytes(5), src=src)
    tier.wait_for_writes()

    # Second write with same hash should be a no-op
    src2 = _make_block(16, seed=6)
    tier.write_block_async(block_hash=to_block_hash_bytes(5), src=src2)
    tier.wait_for_writes()

    # Read should return original data
    dest = np.zeros(16, dtype=np.uint8)
    future = tier.read_block_async(block_hash=to_block_hash_bytes(5), dest=dest)
    future.result()
    np.testing.assert_array_equal(src, dest)

    tier.shutdown()


# -- Block removal --


def test_remove_block(cache_dir: str) -> None:
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        max_disk_size_bytes=10_000,
    )
    src = _make_block(16, seed=11)
    tier.write_block_async(block_hash=to_block_hash_bytes(11), src=src)
    tier.wait_for_writes()
    assert tier.contains(to_block_hash_bytes(11))

    tier.remove(to_block_hash_bytes(11))
    assert not tier.contains(to_block_hash_bytes(11))

    tier.shutdown()


# -- write_block_async() returns Future | None --


def test_returns_future_for_new_block(cache_dir: str) -> None:
    """write_block_async returns a Future when a write is submitted."""
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        max_disk_size_bytes=10_000,
    )
    src = _make_block(16, seed=1)
    result = tier.write_block_async(block_hash=to_block_hash_bytes(1), src=src)
    assert result is not None, "Should return a Future for new block"
    result.result()
    tier.shutdown()


def test_returns_none_for_duplicate_block(cache_dir: str) -> None:
    """write_block_async returns None when the block is already on disk."""
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        max_disk_size_bytes=10_000,
    )
    src = _make_block(16, seed=1)
    tier.write_block_async(block_hash=to_block_hash_bytes(1), src=src)
    tier.wait_for_writes()

    # Second write with same hash → None (already on disk)
    src2 = _make_block(16, seed=2)
    result = tier.write_block_async(block_hash=to_block_hash_bytes(1), src=src2)
    assert result is None, "Should return None for duplicate block"
    tier.shutdown()


def test_returns_none_for_pending_block(cache_dir: str) -> None:
    """write_block_async returns None when a write is already in-flight."""
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        max_disk_size_bytes=10_000,
    )
    src1 = _make_block(16, seed=1)
    first = tier.write_block_async(block_hash=to_block_hash_bytes(1), src=src1)
    assert first is not None

    # While first write is in-flight, submit same hash
    src2 = _make_block(16, seed=2)
    second = tier.write_block_async(block_hash=to_block_hash_bytes(1), src=src2)
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
        max_disk_size_bytes=10_000,
        use_direct_io=True,
    )

    if not has_odirect:
        assert not tier._use_direct_io, "Should fall back to buffered I/O"

    # Regardless of mode, read/write should still work
    src = _make_block(16, seed=42)
    tier.write_block_async(block_hash=to_block_hash_bytes(100), src=src)
    tier.wait_for_writes()

    dest = np.zeros(16, dtype=np.uint8)
    future = tier.read_block_async(
        block_hash=to_block_hash_bytes(100), dest=dest
    )
    future.result()
    np.testing.assert_array_equal(src, dest)

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
        max_disk_size_bytes=10 * 4096,
        use_direct_io=True,
    )

    if not tier._use_direct_io:
        pytest.skip("O_DIRECT disabled due to alignment constraints")

    src = _make_block(4096, seed=99)
    tier.write_block_async(block_hash=to_block_hash_bytes(50), src=src)
    tier.wait_for_writes()

    dest = np.zeros(4096, dtype=np.uint8)
    future = tier.read_block_async(
        block_hash=to_block_hash_bytes(50), dest=dest
    )
    future.result()
    np.testing.assert_array_equal(src, dest)

    tier.shutdown()


def test_rebuild_after_concurrent_writes(cache_dir: str) -> None:
    """A fresh instance rebuilds the index from disk after concurrent writes."""
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        max_disk_size_bytes=10_000,
        num_workers=8,
    )

    num_blocks = 64
    futures = []
    for h in range(num_blocks):
        src = _make_block(16, seed=h)
        f = tier.write_block_async(block_hash=to_block_hash_bytes(h), src=src)
        if f is not None:
            futures.append(f)

    for f in futures:
        f.result()

    for h in range(num_blocks):
        assert tier.contains(to_block_hash_bytes(h))

    tier.shutdown()

    tier2 = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        max_disk_size_bytes=10_000,
    )
    assert tier2.num_used_blocks == num_blocks
    for h in range(num_blocks):
        assert tier2.contains(to_block_hash_bytes(h))
    tier2.shutdown()


def test_scan_ignores_non_block_files(cache_dir: str) -> None:
    """The startup scan skips non-'.bin' and malformed-hash files."""
    tier = DiskTier(
        cache_dir=cache_dir, block_nbytes=16, max_disk_size_bytes=10_000
    )
    tier.write_block_async(
        block_hash=to_block_hash_bytes(1), src=_make_block(16, seed=1)
    )
    tier.wait_for_writes()
    tier.shutdown()

    (Path(cache_dir) / "notes.txt").write_text("junk")
    (Path(cache_dir) / "zzzz.bin").write_text("non-hex stem")

    tier2 = DiskTier(
        cache_dir=cache_dir, block_nbytes=16, max_disk_size_bytes=10_000
    )
    assert tier2.num_used_blocks == 1
    assert tier2.contains(to_block_hash_bytes(1))
    tier2.shutdown()


def test_direct_io_disabled_on_unaligned_blocks(cache_dir: str) -> None:
    """Verify O_DIRECT is disabled when block size is not FS-aligned."""
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=17,  # Not aligned to any FS block size
        max_disk_size_bytes=10_000,
        use_direct_io=True,
    )
    # 17 bytes is not aligned to 4096 (typical FS block size), so
    # O_DIRECT should be disabled automatically.
    assert not tier._use_direct_io, (
        "Should disable O_DIRECT for unaligned block sizes"
    )
    tier.shutdown()


# -- Async eviction (off the scheduler critical path) --


def test_eviction_reclaims_disk_space(cache_dir: str) -> None:
    """Async eviction unlinks files so on-disk usage stays within budget."""
    # Budget for exactly 2 blocks.
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        max_disk_size_bytes=32,
    )
    # Drain between writes so each eviction sees the prior committed blocks
    # (eviction is keyed on the committed index, which updates on completion).
    for h in range(5):
        tier.write_block_async(
            block_hash=to_block_hash_bytes(h), src=_make_block(16, seed=h)
        )
        tier.wait_for_writes()

    # wait_for_writes drains in-flight evictions, so the evicted files are
    # actually gone from disk -- not just dropped from the in-memory index.
    on_disk = list(Path(cache_dir).rglob("*.bin"))
    assert len(on_disk) == 2
    assert tier.num_used_blocks == 2

    tier.shutdown()


def test_rewrite_after_eviction(cache_dir: str) -> None:
    """A hash can be re-written after it was evicted and its delete drained."""
    # Room for a single block.
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        max_disk_size_bytes=16,
    )
    h1 = to_block_hash_bytes(1)
    h2 = to_block_hash_bytes(2)
    assert tier.write_block_async(block_hash=h1, src=_make_block(16, 1))
    tier.wait_for_writes()

    # Writing a second block evicts hash 1.
    assert tier.write_block_async(block_hash=h2, src=_make_block(16, 2))
    tier.wait_for_writes()
    assert not tier.contains(h1)

    # Hash 1's delete has drained, so it can be written again (evicting 2).
    assert tier.write_block_async(block_hash=h1, src=_make_block(16, 1))
    tier.wait_for_writes()
    assert tier.contains(h1)
    assert not tier.contains(h2)

    tier.shutdown()


# -- Directory sharding layout --


def test_blocks_stored_in_shard_subdirs(cache_dir: str) -> None:
    """Blocks live under a per-hash bucket subdir, not the cache root."""
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        max_disk_size_bytes=10_000,
    )
    block_hash = to_block_hash_bytes(0x123)
    tier.write_block_async(block_hash=block_hash, src=_make_block(16, seed=1))
    tier.wait_for_writes()

    # No block files at the root; exactly one under a two-hex bucket directory
    # named for the first byte of the hash.
    assert list(Path(cache_dir).glob("*.bin")) == []
    sharded = list(Path(cache_dir).glob("*/*.bin"))
    assert len(sharded) == 1
    assert sharded[0].parent.name == f"{block_hash[0]:02x}"

    tier.shutdown()


def test_legacy_flat_file_ignored_on_warm_start(cache_dir: str) -> None:
    """A file left by the old flat layout is not indexed (cold start)."""
    tier = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        max_disk_size_bytes=10_000,
    )
    tier.shutdown()

    # Drop a legacy flat-layout file at the cache root (not in a bucket).
    legacy_hash = to_block_hash_bytes(0xABC)
    (Path(cache_dir) / f"{legacy_hash.hex()}.bin").write_bytes(b"\x00" * 16)

    tier2 = DiskTier(
        cache_dir=cache_dir,
        block_nbytes=16,
        max_disk_size_bytes=10_000,
    )
    assert not tier2.contains(legacy_hash)
    assert tier2.num_used_blocks == 0
    tier2.shutdown()
