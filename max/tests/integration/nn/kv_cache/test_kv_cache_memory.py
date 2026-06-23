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
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.nn.kv_cache.cache_params import (
    KVCacheBuffer,
    KVCacheMemory,
    ReplicatedKVCacheMemory,
)


def _buffer(
    num_pages: int,
    bytes_per_page: int,
    dtype: DType = DType.uint8,
    device: Device | None = None,
) -> Buffer:
    """Allocate a 2-D buffer on CPU for memory-layout tests."""
    return Buffer(
        shape=(num_pages, bytes_per_page),
        dtype=dtype,
        device=device if device is not None else CPU(),
    )


def _uint8_buffer(num_pages: int, bytes_per_page: int) -> Buffer:
    """Allocate a 2-D ``uint8`` buffer on CPU for memory-layout tests."""
    return _buffer(num_pages, bytes_per_page)


# ==================== KVCacheMemory Tests ====================


def test_kv_cache_memory_valid_buffer() -> None:
    """A 2-D uint8 buffer passes validation."""
    memory = KVCacheMemory(buffer=_uint8_buffer(4, 256))
    assert memory.total_num_pages == 4


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


def test_replicated_kv_cache_memory_matching_peers() -> None:
    """Peers sharing the same shape as the rank-0 buffer validate."""
    memory = ReplicatedKVCacheMemory(
        buffer=_uint8_buffer(4, 256),
        peers=[_uint8_buffer(4, 256), _uint8_buffer(4, 256)],
    )
    assert len(memory.peers) == 2
    assert all(peer.shape[0] == 4 for peer in memory.peers)


def test_replicated_kv_cache_memory_validates_at_least_one_peer() -> None:
    """A replicated unit with no peers raises ValueError."""
    with pytest.raises(
        ValueError, match="ReplicatedKVCacheMemory must have at least one peer"
    ):
        ReplicatedKVCacheMemory(buffer=_uint8_buffer(4, 256), peers=[])


def test_replicated_kv_cache_memory_validates_peer_buffer() -> None:
    """A peer that is not a 2-D uint8 buffer raises ValueError."""
    peer = Buffer(shape=(4,), dtype=DType.uint8, device=CPU())
    with pytest.raises(
        ValueError, match="KVCacheMemory buffer must have 2 dimensions"
    ):
        ReplicatedKVCacheMemory(buffer=_uint8_buffer(4, 256), peers=[peer])


def test_replicated_kv_cache_memory_mismatched_peer_fails() -> None:
    """A peer with a different shape raises ValueError."""
    with pytest.raises(
        ValueError, match="All buffers must have the same shape"
    ):
        ReplicatedKVCacheMemory(
            buffer=_uint8_buffer(4, 256),
            peers=[_uint8_buffer(4, 512)],
        )


def test_replicated_kv_cache_memory_mismatched_among_peers_fails() -> None:
    """Peers that disagree with each other raise ValueError."""
    with pytest.raises(
        ValueError, match="All buffers must have the same shape"
    ):
        ReplicatedKVCacheMemory(
            buffer=_uint8_buffer(4, 256),
            peers=[_uint8_buffer(4, 256), _uint8_buffer(4, 128)],
        )


# ==================== KVCacheBuffer Tests ====================


def test_kv_cache_buffer_empty_values_fails() -> None:
    """An empty values list raises ValueError before the TP-shard check."""
    with pytest.raises(ValueError, match="List of values must be non-empty"):
        KVCacheBuffer(
            values=[],
            replicates_kv_across_tp=True,
        )


def test_kv_cache_buffer_replicated_single_shard_fails() -> None:
    """replicates_kv_across_tp=True with a single shard raises ValueError."""
    with pytest.raises(ValueError, match="requires at least 2 TP shards"):
        KVCacheBuffer(
            values=[_uint8_buffer(4, 256)],
            replicates_kv_across_tp=True,
        )


def test_kv_cache_buffer_replicated_multi_shard_emits_peers() -> None:
    """replicates_kv_across_tp=True with >=2 shards yields non-empty peers."""
    kv_buffer = KVCacheBuffer(
        values=[_uint8_buffer(4, 256), _uint8_buffer(4, 256)],
        replicates_kv_across_tp=True,
    )
    memory = kv_buffer.to_memory()
    assert len(memory) == 1
    assert isinstance(memory[0], ReplicatedKVCacheMemory)
    assert len(memory[0].peers) == 1


def test_kv_cache_buffer_sharded_single_shard_valid() -> None:
    """A non-replicated single-shard buffer validates and exposes its pages."""
    kv_buffer = KVCacheBuffer(
        values=[_uint8_buffer(4, 256)],
        replicates_kv_across_tp=False,
    )
    assert kv_buffer.total_num_pages == 4
    assert kv_buffer.all_buffers == kv_buffer.values


def test_kv_cache_buffer_mismatched_value_dtypes_fails() -> None:
    """Values that disagree on dtype raise ValueError."""
    with pytest.raises(ValueError, match="All values must have the same dtype"):
        KVCacheBuffer(
            values=[
                _buffer(4, 256, dtype=DType.uint8),
                _buffer(4, 256, dtype=DType.bfloat16),
            ],
            replicates_kv_across_tp=False,
        )


def test_kv_cache_buffer_mismatched_value_shapes_fails() -> None:
    """Values that disagree on shape raise ValueError."""
    with pytest.raises(ValueError, match="All values must have the same shape"):
        KVCacheBuffer(
            values=[_uint8_buffer(4, 256), _uint8_buffer(4, 512)],
            replicates_kv_across_tp=False,
        )


# ==================== KVCacheBuffer + scales Tests ====================


def test_kv_cache_buffer_with_scales_valid() -> None:
    """Values and scales of equal length and page count validate."""
    kv_buffer = KVCacheBuffer(
        values=[_buffer(4, 256, dtype=DType.uint8)],
        scales=[_buffer(4, 8, dtype=DType.float32)],
        replicates_kv_across_tp=False,
    )
    assert kv_buffer.scales is not None
    assert kv_buffer.total_num_pages == 4
    assert kv_buffer.all_buffers == [*kv_buffer.values, *kv_buffer.scales]


def test_kv_cache_buffer_with_scales_to_memory() -> None:
    """to_memory emits one memory unit per value and per scale shard."""
    kv_buffer = KVCacheBuffer(
        values=[_buffer(4, 256, dtype=DType.uint8)],
        scales=[_buffer(4, 8, dtype=DType.float32)],
        replicates_kv_across_tp=False,
    )
    memory = kv_buffer.to_memory()
    assert len(memory) == 2
    assert all(isinstance(m, KVCacheMemory) for m in memory)


def test_kv_cache_buffer_scales_length_mismatch_fails() -> None:
    """A scales list shorter than values raises ValueError."""
    with pytest.raises(
        ValueError, match="Scales must be the same length as values"
    ):
        KVCacheBuffer(
            values=[_uint8_buffer(4, 256), _uint8_buffer(4, 256)],
            scales=[_buffer(4, 8, dtype=DType.float32)],
            replicates_kv_across_tp=False,
        )


def test_kv_cache_buffer_mismatched_scale_dtypes_fails() -> None:
    """Scales that disagree on dtype raise ValueError."""
    with pytest.raises(ValueError, match="All scales must have the same dtype"):
        KVCacheBuffer(
            values=[_uint8_buffer(4, 256), _uint8_buffer(4, 256)],
            scales=[
                _buffer(4, 8, dtype=DType.float32),
                _buffer(4, 8, dtype=DType.bfloat16),
            ],
            replicates_kv_across_tp=False,
        )


def test_kv_cache_buffer_mismatched_scale_shapes_fails() -> None:
    """Scales that disagree on shape raise ValueError."""
    with pytest.raises(ValueError, match="All scales must have the same shape"):
        KVCacheBuffer(
            values=[_uint8_buffer(4, 256), _uint8_buffer(4, 256)],
            scales=[
                _buffer(4, 8, dtype=DType.float32),
                _buffer(4, 16, dtype=DType.float32),
            ],
            replicates_kv_across_tp=False,
        )


def test_kv_cache_buffer_value_scale_page_count_mismatch_fails() -> None:
    """Values and scales with differing page counts raise ValueError."""
    with pytest.raises(
        ValueError, match="Values and scales must have the same number of pages"
    ):
        KVCacheBuffer(
            values=[_buffer(4, 256, dtype=DType.uint8)],
            scales=[_buffer(8, 8, dtype=DType.float32)],
            replicates_kv_across_tp=False,
        )
