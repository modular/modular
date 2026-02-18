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

"""Tests for LocalConnector KV cache host memory offloading."""

import numpy as np
import pytest
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.graph import DeviceRef
from max.kv_cache.connectors.local_connector import LocalConnector
from max.nn.legacy.kv_cache import KVCacheParams
from test_common.context_utils import create_text_context


def create_local_connector(
    num_device_blocks: int = 64,
    num_host_blocks: int = 32,
    page_size: int = 16,
    num_layers: int = 2,
    n_kv_heads: int = 4,
    head_dim: int = 64,
) -> LocalConnector:
    """Create a LocalConnector for testing.

    Creates the minimal device tensors needed for the connector to function,
    without creating the full PagedKVCacheManager infrastructure.
    """
    if accelerator_count() == 0:
        pytest.skip("No GPU available")

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

    # Create device tensors required by the connector
    device_tensors = [
        Buffer(
            shape=[num_device_blocks, *kv_params.shape_per_block],
            dtype=kv_params.dtype,
            device=device,
        )
    ]

    return LocalConnector(
        params=kv_params,
        devices=[device],
        device_tensors=device_tensors,
        device_scale_tensors=None,
        total_num_host_blocks=num_host_blocks,
    )


def test_connector_name() -> None:
    """Verify LocalConnector has correct name."""
    connector = create_local_connector()
    assert connector.name == "LocalConnector"


def test_host_tensors_are_pinned() -> None:
    """Verify host tensors are pinned for efficient DMA transfers."""
    connector = create_local_connector()
    assert connector.host_tensors is not None

    for tensor in connector.host_tensors:
        assert tensor.pinned, "Host tensors should be pinned memory"


def test_num_host_blocks() -> None:
    """Verify num_host_blocks returns the configured value."""
    num_host = 48
    connector = create_local_connector(num_host_blocks=num_host)
    assert connector.num_host_blocks == num_host


def test_num_used_host_blocks_initially_zero() -> None:
    """Verify no host blocks are used initially."""
    connector = create_local_connector()
    assert connector.num_used_host_blocks == 0


def test_save_queues_blocks_for_offload() -> None:
    """Verify save() queues blocks without immediate transfer."""
    connector = create_local_connector()

    connector.save([0, 1, 2], [100, 200, 300])

    assert len(connector._pending_saves) == 3
    assert connector.num_used_host_blocks == 0


def test_flush_executes_pending_saves() -> None:
    """Verify flush() transfers queued blocks to host cache."""
    connector = create_local_connector()

    connector.save([0, 1], [100, 200])
    connector.flush()

    assert len(connector._pending_saves) == 0
    assert connector.num_used_host_blocks == 2


def test_duplicate_hash_not_saved_twice() -> None:
    """Verify blocks with same hash are deduplicated."""
    connector = create_local_connector()

    connector.save([0], [100])
    connector.flush()
    connector.save([1], [100])
    connector.flush()

    assert connector.num_used_host_blocks == 1


def test_lookup_returns_zero_for_empty_cache() -> None:
    """Verify lookup returns 0 tokens when cache is empty."""
    connector = create_local_connector(page_size=16)

    ctx = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    tokens = connector.lookup(ctx, [100, 200, 300])

    assert tokens == 0


def test_lookup_finds_cached_blocks() -> None:
    """Verify lookup returns correct token count for cached blocks."""
    page_size = 16
    connector = create_local_connector(page_size=page_size)

    connector.save([0, 1, 2], [100, 200, 300])
    connector.flush()

    ctx = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    tokens = connector.lookup(ctx, [100, 200, 300])

    assert tokens == 3 * page_size


def test_lookup_stops_at_first_miss() -> None:
    """Verify lookup returns contiguous prefix only."""
    page_size = 16
    connector = create_local_connector(page_size=page_size)

    connector.save([0], [100])
    connector.save([2], [300])
    connector.flush()

    ctx = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    tokens = connector.lookup(ctx, [100, 200, 300])

    assert tokens == 1 * page_size


def test_load_returns_hashes_for_loaded_blocks() -> None:
    """Verify load() returns block hashes for H2D transfers."""
    page_size = 16
    connector = create_local_connector(page_size=page_size)

    connector.save([0, 1], [100, 200])
    connector.flush()

    ctx = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    connector.lookup(ctx, [100, 200])

    loaded_hashes = connector.load(ctx, [10, 11], [])

    assert loaded_hashes == [100, 200]


def test_load_without_lookup_returns_empty() -> None:
    """Verify load() returns empty list without prior lookup."""
    connector = create_local_connector()

    ctx = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    loaded_hashes = connector.load(ctx, [0, 1], [])

    assert loaded_hashes == []


def test_multiple_contexts_have_independent_pending_loads() -> None:
    """Verify each context has its own pending load state."""
    page_size = 16
    connector = create_local_connector(page_size=page_size)

    connector.save([0, 1, 2, 3], [100, 200, 300, 400])
    connector.flush()

    ctx1 = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    ctx2 = create_text_context(np.array([4, 5, 6], dtype=np.int64))

    tokens1 = connector.lookup(ctx1, [100, 200])
    tokens2 = connector.lookup(ctx2, [300, 400])

    assert tokens1 == 2 * page_size
    assert tokens2 == 2 * page_size
    assert str(ctx1.request_id) in connector._pending_loads
    assert str(ctx2.request_id) in connector._pending_loads


def test_load_for_one_context_does_not_affect_another() -> None:
    """Verify loading blocks for one context doesn't consume another's pending loads."""
    page_size = 16
    connector = create_local_connector(page_size=page_size)

    connector.save([0, 1, 2, 3], [100, 200, 300, 400])
    connector.flush()

    ctx1 = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    ctx2 = create_text_context(np.array([4, 5, 6], dtype=np.int64))

    connector.lookup(ctx1, [100, 200])
    connector.lookup(ctx2, [300, 400])

    loaded1 = connector.load(ctx1, [10, 11], [])
    assert loaded1 == [100, 200]

    assert str(ctx2.request_id) in connector._pending_loads
    loaded2 = connector.load(ctx2, [12, 13], [])
    assert loaded2 == [300, 400]


def test_on_request_complete_only_affects_target_context() -> None:
    """Verify completing one request doesn't affect another's pending loads."""
    page_size = 16
    connector = create_local_connector(page_size=page_size)

    connector.save([0, 1], [100, 200])
    connector.flush()

    ctx1 = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    ctx2 = create_text_context(np.array([4, 5, 6], dtype=np.int64))

    connector.lookup(ctx1, [100])
    connector.lookup(ctx2, [200])

    connector.on_request_complete(ctx1.request_id, [0])

    assert str(ctx1.request_id) not in connector._pending_loads
    assert str(ctx2.request_id) in connector._pending_loads

    loaded2 = connector.load(ctx2, [10], [])
    assert loaded2 == [200]


def test_prefix_cache_hit_full_sequence() -> None:
    """Verify full prefix cache hit: save -> lookup -> load round-trip."""
    page_size = 16
    connector = create_local_connector(page_size=page_size)

    connector.save([0, 1, 2], [100, 200, 300])
    connector.flush()
    assert connector.num_used_host_blocks == 3

    ctx = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    tokens = connector.lookup(ctx, [100, 200, 300])
    assert tokens == 3 * page_size

    loaded_hashes = connector.load(ctx, [10, 11, 12], [])
    assert loaded_hashes == [100, 200, 300]

    assert connector.num_used_host_blocks == 3


def test_prefix_cache_partial_hit() -> None:
    """Verify partial prefix cache hit returns only matching prefix."""
    page_size = 16
    connector = create_local_connector(page_size=page_size)

    connector.save([0, 1], [100, 200])
    connector.flush()

    ctx = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    tokens = connector.lookup(ctx, [100, 200, 300])
    assert tokens == 2 * page_size

    loaded_hashes = connector.load(ctx, [10, 11], [])
    assert loaded_hashes == [100, 200]


def test_prefix_cache_miss_at_start() -> None:
    """Verify cache miss at start of sequence returns nothing."""
    page_size = 16
    connector = create_local_connector(page_size=page_size)

    connector.save([1, 2], [200, 300])
    connector.flush()

    ctx = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    tokens = connector.lookup(ctx, [100, 200, 300])
    assert tokens == 0

    loaded_hashes = connector.load(ctx, [], [])
    assert loaded_hashes == []


def test_on_request_complete_clears_pending_loads() -> None:
    """Verify on_request_complete clears pending load state."""
    page_size = 16
    connector = create_local_connector(page_size=page_size)

    connector.save([0], [100])
    connector.flush()

    ctx = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    connector.lookup(ctx, [100])

    assert str(ctx.request_id) in connector._pending_loads

    connector.on_request_complete(ctx.request_id, [0])

    assert str(ctx.request_id) not in connector._pending_loads


def test_reset_prefix_cache_clears_host_cache() -> None:
    """Verify reset_prefix_cache clears all cached blocks."""
    connector = create_local_connector()

    connector.save([0, 1, 2], [100, 200, 300])
    connector.flush()
    assert connector.num_used_host_blocks == 3

    connector.reset_prefix_cache()

    assert connector.num_used_host_blocks == 0


def test_shutdown_clears_pending_state() -> None:
    """Verify shutdown clears state and waits for transfers."""
    connector = create_local_connector()

    connector.save([0, 1], [100, 200])

    connector.shutdown()

    assert len(connector._pending_saves) == 0
    assert len(connector._pending_loads) == 0
