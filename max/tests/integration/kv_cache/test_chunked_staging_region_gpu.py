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

"""Tests for ``_ChunkedStagingRegion``, the host KV cache's backing store.

Being registered in chunks, it promises less than a ``Buffer``: an address, a
lifetime, and that no chunk boundary splits a row. These pin all three.

That the wrap makes pages DMA-capable is covered a layer down, in
``MLRT/unittests/Driver/GPUDeviceContextTest.cpp``.
"""

import ctypes
import resource
import sys
from itertools import pairwise

import pytest
from max.driver import (
    CPU,
    Accelerator,
    _ChunkedStagingRegion,
    accelerator_count,
)

pytestmark = pytest.mark.skipif(
    accelerator_count() == 0, reason="No GPU available"
)

BLOCK_BYTES = 1024 * 1024


def _page() -> int:
    return ctypes.CDLL(None).getpagesize()


def _region(
    blocks: int, block_bytes: int = BLOCK_BYTES, **kw: int
) -> _ChunkedStagingRegion:
    """A region shaped as the host KV cache allocates one."""
    return _ChunkedStagingRegion(
        byte_size=blocks * block_bytes,
        row_bytes=block_bytes,
        device=Accelerator(),
        **kw,
    )


def _block_view(region: _ChunkedStagingRegion, bid: int) -> memoryview:
    """Block ``bid``, addressed the way a consumer addresses it."""
    addr = region.address + bid * region.row_bytes
    return memoryview(
        (ctypes.c_uint8 * region.row_bytes).from_address(addr)
    ).cast("B")


# --- shape and sizing ----------------------------------------------------- #


def test_reports_its_geometry() -> None:
    region = _region(8)
    assert region.address != 0
    assert region.byte_size == 8 * BLOCK_BYTES
    assert region.row_bytes == BLOCK_BYTES
    assert region.num_chunks >= 1


def test_chunk_size_never_splits_a_row_or_a_page() -> None:
    """The invariant the chunking scheme exists to preserve."""
    region = _region(64, chunk_bytes=3 * BLOCK_BYTES + 7)
    assert region.chunk_bytes % region.row_bytes == 0
    assert region.chunk_bytes % _page() == 0


def test_chunk_count_follows_chunk_size() -> None:
    region = _region(64, chunk_bytes=8 * BLOCK_BYTES)
    assert region.chunk_bytes == 8 * BLOCK_BYTES
    assert region.num_chunks == 8


def test_chunk_size_is_rounded_up_never_down_to_zero() -> None:
    """A chunk request below one row still yields whole rows."""
    region = _region(4, chunk_bytes=1)
    assert region.chunk_bytes >= region.row_bytes
    assert region.num_chunks <= 4


def test_single_chunk_when_request_exceeds_the_region() -> None:
    region = _region(4, chunk_bytes=1024 * BLOCK_BYTES)
    assert region.num_chunks == 1


# --- the contract the offload engine depends on --------------------------- #


def test_block_indexed_access_pattern() -> None:
    """Stands in for the offload engine, the only production consumer.

    It gets a base address and a row width, then copies block ``bid`` at
    ``base + bid * row_bytes`` one block at a time. Each span must be
    writable, distinct, and inside one chunk, since a transfer crossing a
    boundary fails.
    """
    blocks = 64
    region = _region(blocks, chunk_bytes=5 * BLOCK_BYTES)
    assert region.num_chunks > 1, "the interesting case is more than one chunk"

    host_base, row = region.address, region.row_bytes
    for bid in range(blocks):
        start = bid * row
        first_chunk = start // region.chunk_bytes
        last_chunk = (start + row - 1) // region.chunk_bytes
        assert first_chunk == last_chunk, (
            f"block {bid} spans chunks {first_chunk}..{last_chunk}; a copy of "
            "it would cross a registration boundary"
        )
        assert host_base + start == region.address + bid * row

    for bid in range(blocks):
        _block_view(region, bid)[:] = bytes([bid % 256]) * row
    for bid in range(blocks):
        block = _block_view(region, bid)
        assert block[0] == bid % 256
        assert block[-1] == bid % 256


def test_blocks_are_laid_out_back_to_back() -> None:
    """Every block address derives from one base, so the run is gap-free."""
    region = _region(4)
    addrs = [region.address + bid * region.row_bytes for bid in range(4)]
    gaps = [b - a for a, b in pairwise(addrs)]
    assert gaps == [BLOCK_BYTES] * 3


# --- lifetime ------------------------------------------------------------- #


def _peak_rss_bytes() -> int:
    """Peak RSS. ru_maxrss is bytes on macOS, KiB on Linux."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def test_release_unmaps_rather_than_leaking() -> None:
    """Every region is faulted in, so a skipped munmap shows up in peak RSS."""
    region_bytes = 64 * 1024 * 1024
    cycles = 8
    before = _peak_rss_bytes()
    for _ in range(cycles):
        region = _ChunkedStagingRegion(
            byte_size=region_bytes,
            row_bytes=8 * 1024 * 1024,
            device=Accelerator(),
        )
        _block_view(region, 0)[:1024] = b"\x09" * 1024
        del region
    growth = _peak_rss_bytes() - before
    # Leaking every region costs cycles * region_bytes; a release keeps one.
    assert growth < 3 * region_bytes, (
        f"peak RSS grew {growth / 1024**2:.0f} MiB over {cycles} x "
        f"{region_bytes / 1024**2:.0f} MiB cycles; the mapping is not released"
    )


# --- rejected inputs ------------------------------------------------------ #


def test_rejects_cpu_device() -> None:
    with pytest.raises(ValueError, match="non-host device"):
        _ChunkedStagingRegion(
            byte_size=BLOCK_BYTES, row_bytes=BLOCK_BYTES, device=CPU()
        )


def test_rejects_zero_byte_size() -> None:
    with pytest.raises(ValueError, match="byte_size"):
        _ChunkedStagingRegion(
            byte_size=0, row_bytes=BLOCK_BYTES, device=Accelerator()
        )


def test_rejects_zero_row_bytes() -> None:
    with pytest.raises(ValueError, match="row_bytes"):
        _ChunkedStagingRegion(
            byte_size=BLOCK_BYTES, row_bytes=0, device=Accelerator()
        )


def test_rejects_byte_size_that_is_not_whole_rows() -> None:
    with pytest.raises(ValueError, match="not a multiple of row_bytes"):
        _ChunkedStagingRegion(
            byte_size=BLOCK_BYTES + 1,
            row_bytes=BLOCK_BYTES,
            device=Accelerator(),
        )


def test_rejects_zero_chunk_bytes() -> None:
    with pytest.raises(ValueError, match="chunk_bytes"):
        _ChunkedStagingRegion(
            byte_size=BLOCK_BYTES,
            row_bytes=BLOCK_BYTES,
            device=Accelerator(),
            chunk_bytes=0,
        )
