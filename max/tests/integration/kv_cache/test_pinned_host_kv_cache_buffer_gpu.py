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

"""Tests for PinnedHostKVCacheBuffer chunked pinned allocation."""

import numpy as np
import pytest
from max.driver import Accelerator, accelerator_count
from max.pipelines.kv_cache.paged_kv_cache.block_copy_engine import (
    PinnedHostKVCacheBuffer,
)

pytestmark = pytest.mark.skipif(
    accelerator_count() == 0, reason="No GPU available"
)

BYTES_PER_BLOCK = 1024


_DEFAULT_MAX_CHUNK_BYTES = 4 * BYTES_PER_BLOCK


def _make_buffer(
    total_num_blocks: int,
    bytes_per_block: int = BYTES_PER_BLOCK,
    max_chunk_bytes: int = _DEFAULT_MAX_CHUNK_BYTES,
) -> PinnedHostKVCacheBuffer:
    return PinnedHostKVCacheBuffer(
        total_num_blocks=total_num_blocks,
        bytes_per_block=bytes_per_block,
        device=Accelerator(),
        max_chunk_bytes=max_chunk_bytes,
    )


def test_single_chunk_when_small() -> None:
    buf = _make_buffer(3)
    assert len(buf._chunks) == 1
    assert buf.shape == [3, BYTES_PER_BLOCK]
    assert buf.pinned


def test_multiple_chunks() -> None:
    buf = _make_buffer(5, max_chunk_bytes=2 * BYTES_PER_BLOCK)
    assert len(buf._chunks) == 3
    assert buf.shape == [5, BYTES_PER_BLOCK]


def test_locate_oob_too_large() -> None:
    buf = _make_buffer(10)
    with pytest.raises(IndexError, match="out of range"):
        buf._locate(10)


def test_page_view_returns_correct_size() -> None:
    buf = _make_buffer(4)
    view = buf.page_view(0)
    assert view.num_elements == BYTES_PER_BLOCK


def test_numpy_page_view_shape() -> None:
    buf = _make_buffer(4)
    arr = buf.numpy_page_view(0)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (BYTES_PER_BLOCK,)
    assert arr.dtype == np.uint8


def test_numpy_page_view_is_writable() -> None:
    buf = _make_buffer(4)
    arr = buf.numpy_page_view(0)
    arr[:] = 42
    readback = buf.numpy_page_view(0)
    assert np.all(readback == 42)


def test_pages_are_independent() -> None:
    buf = _make_buffer(4)
    buf.numpy_page_view(0)[:] = 1
    buf.numpy_page_view(1)[:] = 2
    assert np.all(buf.numpy_page_view(0) == 1)
    assert np.all(buf.numpy_page_view(1) == 2)


def test_pages_independent_across_chunks() -> None:
    buf = _make_buffer(4, max_chunk_bytes=2 * BYTES_PER_BLOCK)
    assert len(buf._chunks) == 2
    buf.numpy_page_view(0)[:] = 10
    buf.numpy_page_view(2)[:] = 20
    assert np.all(buf.numpy_page_view(0) == 10)
    assert np.all(buf.numpy_page_view(2) == 20)
