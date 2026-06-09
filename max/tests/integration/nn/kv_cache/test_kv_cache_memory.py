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

import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.nn.kv_cache.cache_params import (
    KVCacheMemory,
    ReplicatedKVCacheMemory,
)


def _uint8_buffer(num_pages: int, bytes_per_page: int) -> Buffer:
    """Allocate a 2-D ``uint8`` buffer on CPU for memory-layout tests."""
    return Buffer(
        shape=(num_pages, bytes_per_page),
        dtype=DType.uint8,
        device=CPU(),
    )


# ==================== KVCacheMemory Tests ====================


def test_kv_cache_memory_valid_buffer() -> None:
    """A 2-D uint8 buffer passes validation."""
    memory = KVCacheMemory(buffer=_uint8_buffer(4, 256))
    assert memory.buffer.shape[1] == 256


def test_kv_cache_memory_non_2d_buffer_fails() -> None:
    """A buffer that is not 2-D raises ValueError."""
    buffer = Buffer(shape=(4,), dtype=DType.uint8, device=CPU())
    with pytest.raises(
        ValueError, match="KVCacheMemory buffer must have 2 dimensions"
    ):
        KVCacheMemory(buffer=buffer)


def test_kv_cache_memory_non_uint8_buffer_fails() -> None:
    """A buffer with a non-uint8 dtype raises ValueError."""
    buffer = Buffer(shape=(4, 256), dtype=DType.bfloat16, device=CPU())
    with pytest.raises(
        ValueError, match="KVCacheMemory buffer must have dtype uint8"
    ):
        KVCacheMemory(buffer=buffer)


# ==================== ReplicatedKVCacheMemory Tests ====================


def test_replicated_kv_cache_memory_no_peers() -> None:
    """A replicated unit with no peers always has consistent bytes_per_page."""
    memory = ReplicatedKVCacheMemory(buffer=_uint8_buffer(4, 256))
    assert memory.peers == []
    assert memory.buffer.shape[1] == 256


def test_replicated_kv_cache_memory_matching_peers() -> None:
    """Peers sharing the same bytes_per_page as the rank-0 buffer validate."""
    memory = ReplicatedKVCacheMemory(
        buffer=_uint8_buffer(4, 256),
        peers=[_uint8_buffer(4, 256), _uint8_buffer(8, 256)],
    )
    assert len(memory.peers) == 2
    assert all(peer.shape[1] == 256 for peer in memory.peers)


def test_replicated_kv_cache_memory_validates_rank0_buffer() -> None:
    """The rank-0 buffer is validated via super().__post_init__()."""
    buffer = Buffer(shape=(4,), dtype=DType.uint8, device=CPU())
    with pytest.raises(
        ValueError, match="KVCacheMemory buffer must have 2 dimensions"
    ):
        ReplicatedKVCacheMemory(buffer=buffer)


def test_replicated_kv_cache_memory_validates_peer_buffer() -> None:
    """A peer that is not a 2-D uint8 buffer raises ValueError."""
    peer = Buffer(shape=(4,), dtype=DType.uint8, device=CPU())
    with pytest.raises(
        ValueError, match="KVCacheMemory buffer must have 2 dimensions"
    ):
        ReplicatedKVCacheMemory(buffer=_uint8_buffer(4, 256), peers=[peer])


def test_replicated_kv_cache_memory_mismatched_peer_fails() -> None:
    """A peer with a different bytes_per_page raises ValueError."""
    with pytest.raises(
        ValueError, match="All buffers must have the same bytes_per_page"
    ):
        ReplicatedKVCacheMemory(
            buffer=_uint8_buffer(4, 256),
            peers=[_uint8_buffer(4, 512)],
        )


def test_replicated_kv_cache_memory_mismatched_among_peers_fails() -> None:
    """Peers that disagree with each other raise ValueError."""
    with pytest.raises(
        ValueError, match="All buffers must have the same bytes_per_page"
    ):
        ReplicatedKVCacheMemory(
            buffer=_uint8_buffer(4, 256),
            peers=[_uint8_buffer(4, 256), _uint8_buffer(4, 128)],
        )
