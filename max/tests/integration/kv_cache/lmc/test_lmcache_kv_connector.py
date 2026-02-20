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

"""Integration tests for LMCacheConnector with PagedKVCacheManager.

These tests verify the LMCache connector works correctly when integrated
with the full PagedKVCacheManager infrastructure, including:

1. **Connector creation**: LMCacheConnector is correctly instantiated
   when lmcache_config_file is configured.

2. **Save/Flush operations**: KV blocks are correctly saved to LMCache.

3. **Lookup operations**: Cached blocks are found via prefix matching.

4. **Load operations**: Cached blocks are correctly loaded back to device.

5. **Round-trip integrity**: Data survives save -> lookup -> load cycles.

6. **Integration with BlockManager**: Prefix caching works with LMCache backend.

Unlike test_max_gpu_connector.py which tests MAXGPUConnector directly
with flat buffers, these tests use the full paged KV cache infrastructure.

Note: LMCache has a bug where its observability.log_worker thread doesn't
exit cleanly during shutdown. We use vLLM's pattern of wrapping shutdown
in a ThreadPoolExecutor with timeout to avoid hanging.
"""

from __future__ import annotations

import numpy as np
from max.driver import Buffer
from max.kv_cache.connectors.lmcache_connector import LMCacheConnector
from max.kv_cache.paged_kv_cache.cache_manager import PagedKVCacheManager
from max.kv_cache.paged_kv_cache.tp_cache_manager import (
    _TPPagedKVCacheManager,
)
from test_common.context_utils import create_text_context

from .conftest import (
    INTEGRATION_PAGE_SIZE,
    KVCacheTestConfig,
    fill_paged_cache,
    make_dummy_context,
)

# Type alias for the kv_cache_manager fixture yield type.
_ManagerPair = tuple[PagedKVCacheManager, _TPPagedKVCacheManager]


def _get_connector(tp_mgr: _TPPagedKVCacheManager) -> LMCacheConnector:
    """Extract and type-narrow the LMCacheConnector from a tp manager."""
    connector = tp_mgr.connector
    assert isinstance(connector, LMCacheConnector)
    return connector


def get_cache_keys(
    connector: LMCacheConnector,
    block_hashes: list[int],
    block_size: int,
) -> list:
    """Get CacheEngineKey objects from block hashes."""
    offsets = [block_size] * len(block_hashes)
    keys = []
    for _start, _end, key in connector._engine.token_database.process_tokens(
        hashes=block_hashes,
        offsets=offsets,
    ):
        keys.append(key)
    return keys


def query_backend(
    connector: LMCacheConnector,
    block_hashes: list[int],
    block_size: int,
    backend_name: str,
) -> int:
    """Query a specific storage backend to check how many blocks are stored."""
    keys = get_cache_keys(connector, block_hashes, block_size)
    if not keys:
        return 0

    storage_manager = connector._engine.storage_manager
    assert storage_manager is not None

    hit_count, _block_mapping = storage_manager.batched_contains(
        keys=keys,
        search_range=[backend_name],
    )
    return hit_count


# --- Connector creation ---


def test_creates_lmcache_connector(
    kv_cache_manager: _ManagerPair,
) -> None:
    _, tp_mgr = kv_cache_manager
    connector = _get_connector(tp_mgr)
    assert isinstance(connector, LMCacheConnector)
    assert connector.name == "LMCacheConnector"


def test_connector_has_gpu_connector(
    kv_cache_manager: _ManagerPair,
) -> None:
    _, tp_mgr = kv_cache_manager
    connector = _get_connector(tp_mgr)
    assert connector._gpu_connector is not None


# --- Save / Flush ---


def test_save_queues_blocks(
    kv_cache_manager: _ManagerPair,
) -> None:
    _, tp_mgr = kv_cache_manager
    connector = _get_connector(tp_mgr)
    connector.save([0, 1, 2], [100, 200, 300])
    assert len(connector._pending_saves) == 3


def test_flush_executes_pending_saves(
    kv_cache_manager: _ManagerPair,
) -> None:
    _, tp_mgr = kv_cache_manager
    connector = _get_connector(tp_mgr)
    connector.save([0, 1], [100, 200])
    connector.flush()
    assert len(connector._pending_saves) == 0


def test_duplicate_hash_not_saved_twice(
    kv_cache_manager: _ManagerPair,
) -> None:
    _, tp_mgr = kv_cache_manager
    connector = _get_connector(tp_mgr)

    # Save block with hash 100
    connector.save([0], [100])
    connector.flush()

    # Try to save another block with same hash
    connector.save([1], [100])
    connector.flush()

    # Lookup should find exactly 1 block's worth of tokens (not 2)
    ctx = make_dummy_context()
    tokens = connector.lookup(ctx, [100])
    assert tokens == 1 * INTEGRATION_PAGE_SIZE


# --- Lookup ---


def test_lookup_returns_zero_for_uncached_blocks(
    kv_cache_manager: _ManagerPair,
) -> None:
    """Uses unique hash values not used by other tests."""
    _, tp_mgr = kv_cache_manager
    connector = _get_connector(tp_mgr)
    ctx = make_dummy_context()
    tokens = connector.lookup(ctx, [99999, 99998, 99997])
    assert tokens == 0


def test_lookup_finds_cached_blocks(
    kv_cache_manager: _ManagerPair,
) -> None:
    _, tp_mgr = kv_cache_manager
    connector = _get_connector(tp_mgr)
    connector.save([0, 1, 2], [100, 200, 300])
    connector.flush()

    ctx = make_dummy_context()
    tokens = connector.lookup(ctx, [100, 200, 300])
    assert tokens == 3 * INTEGRATION_PAGE_SIZE


def test_lookup_stops_at_first_miss(
    kv_cache_manager: _ManagerPair,
) -> None:
    """Uses unique hashes to ensure the middle block is truly missing."""
    _, tp_mgr = kv_cache_manager
    connector = _get_connector(tp_mgr)
    connector.save([0], [1000])
    connector.save([2], [1002])
    connector.flush()

    ctx = make_dummy_context()
    tokens = connector.lookup(ctx, [1000, 1001, 1002])
    assert tokens == 1 * INTEGRATION_PAGE_SIZE


# --- Load ---


def test_load_returns_hashes_for_loaded_blocks(
    kv_cache_manager: _ManagerPair,
) -> None:
    _, tp_mgr = kv_cache_manager
    connector = _get_connector(tp_mgr)
    connector.save([0, 1], [100, 200])
    connector.flush()

    ctx = make_dummy_context()
    connector.lookup(ctx, [100, 200])
    loaded_hashes = connector.load(ctx, [10, 11], [])
    assert loaded_hashes == [100, 200]


def test_load_without_lookup_returns_empty(
    kv_cache_manager: _ManagerPair,
) -> None:
    _, tp_mgr = kv_cache_manager
    connector = _get_connector(tp_mgr)
    ctx = make_dummy_context()
    loaded_hashes = connector.load(ctx, [0, 1], [])
    assert loaded_hashes == []


# --- Round-trip ---


def test_full_prefix_cache_hit(
    kv_cache_manager: _ManagerPair,
) -> None:
    _, tp_mgr = kv_cache_manager
    connector = _get_connector(tp_mgr)

    # Fill device tensor with pattern so we can verify data integrity
    device_tensor = tp_mgr.device_tensors[0]
    pattern = np.arange(device_tensor.to_numpy().size, dtype=np.float32)
    pattern = pattern.reshape(device_tensor.shape)
    device_tensor.inplace_copy_from(Buffer.from_numpy(pattern))

    connector.save([0, 1, 2], [100, 200, 300])
    connector.flush()

    ctx = make_dummy_context()
    tokens = connector.lookup(ctx, [100, 200, 300])
    assert tokens == 3 * INTEGRATION_PAGE_SIZE

    loaded_hashes = connector.load(ctx, [10, 11, 12], [])
    assert loaded_hashes == [100, 200, 300]


def test_partial_prefix_hit(
    kv_cache_manager: _ManagerPair,
) -> None:
    """Uses unique hashes to ensure the third block is truly missing."""
    _, tp_mgr = kv_cache_manager
    connector = _get_connector(tp_mgr)
    connector.save([0, 1], [2000, 2001])
    connector.flush()

    ctx = make_dummy_context()
    tokens = connector.lookup(ctx, [2000, 2001, 2002])
    assert tokens == 2 * INTEGRATION_PAGE_SIZE

    loaded_hashes = connector.load(ctx, [10, 11], [])
    assert loaded_hashes == [2000, 2001]


def test_multiple_requests_independent(
    kv_cache_manager: _ManagerPair,
) -> None:
    _, tp_mgr = kv_cache_manager
    connector = _get_connector(tp_mgr)
    connector.save([0, 1, 2, 3], [100, 200, 300, 400])
    connector.flush()

    ctx1 = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    ctx2 = create_text_context(np.array([4, 5, 6], dtype=np.int64))

    tokens1 = connector.lookup(ctx1, [100, 200])
    tokens2 = connector.lookup(ctx2, [300, 400])
    assert tokens1 == 2 * INTEGRATION_PAGE_SIZE
    assert tokens2 == 2 * INTEGRATION_PAGE_SIZE

    loaded1 = connector.load(ctx1, [10, 11], [])
    loaded2 = connector.load(ctx2, [12, 13], [])
    assert loaded1 == [100, 200]
    assert loaded2 == [300, 400]


# --- Manager integration ---


def test_claim_alloc_step_with_lmcache(
    kv_cache_manager: _ManagerPair,
) -> None:
    manager, tp_mgr = kv_cache_manager

    tokens = np.arange(INTEGRATION_PAGE_SIZE * 2, dtype=np.int64)
    ctx = create_text_context(tokens)

    manager.claim(ctx.request_id, replica_idx=0)
    manager.alloc(ctx, replica_idx=0, num_steps=1)

    allocated_blocks = tp_mgr.block_manager.req_to_blocks.get(
        ctx.request_id, []
    )
    assert len(allocated_blocks) >= 2

    manager.step([[ctx]])
    manager.release(ctx.request_id, replica_idx=0)


def test_prefix_cache_reuse_with_lmcache(
    kv_cache_manager: _ManagerPair,
) -> None:
    manager, tp_mgr = kv_cache_manager
    connector = _get_connector(tp_mgr)

    # First request - generates and saves blocks
    tokens1 = np.arange(INTEGRATION_PAGE_SIZE, dtype=np.int64)
    ctx1 = create_text_context(tokens1)

    manager.claim(ctx1.request_id, replica_idx=0)
    manager.alloc(ctx1, replica_idx=0, num_steps=1)
    manager.step([[ctx1]])

    list(tp_mgr.block_manager.req_to_blocks.get(ctx1.request_id, []))

    connector.flush()
    manager.release(ctx1.request_id, replica_idx=0)

    # Second request with same prefix - should reuse from cache
    tokens2 = np.arange(INTEGRATION_PAGE_SIZE, dtype=np.int64)
    ctx2 = create_text_context(tokens2)

    manager.claim(ctx2.request_id, replica_idx=0)
    manager.alloc(ctx2, replica_idx=0, num_steps=1)

    block_ids2 = list(
        tp_mgr.block_manager.req_to_blocks.get(ctx2.request_id, [])
    )
    assert len(block_ids2) >= 1

    manager.release(ctx2.request_id, replica_idx=0)


# --- Cleanup ---


def test_on_request_complete_clears_state(
    kv_cache_manager: _ManagerPair,
) -> None:
    _, tp_mgr = kv_cache_manager
    connector = _get_connector(tp_mgr)
    connector.save([0], [100])
    connector.flush()

    ctx = make_dummy_context()
    connector.lookup(ctx, [100])
    assert str(ctx.request_id) in connector._pending_loads

    connector.on_request_complete(ctx.request_id, [0])
    assert str(ctx.request_id) not in connector._pending_loads


def test_flush_clears_pending_and_shutdown_flag_initialized(
    kv_cache_manager: _ManagerPair,
) -> None:
    """Verifies flush() and _is_shutdown initialization.

    We don't call shutdown() because it hangs due to LMCache's
    observability thread bug and would break subsequent tests sharing
    the module-scoped manager.
    """
    _, tp_mgr = kv_cache_manager
    connector = _get_connector(tp_mgr)
    connector.save([0, 1], [100, 200])
    assert len(connector._pending_saves) == 2

    connector.flush()
    assert len(connector._pending_saves) == 0

    assert hasattr(connector, "_is_shutdown")
    assert not connector._is_shutdown


# --- Tiered storage (CPU + Disk) ---


def test_tiered_storage_config(
    kv_cache_manager_with_disk: _TPPagedKVCacheManager,
) -> None:
    """Verify both LocalCPUBackend and LocalDiskBackend are initialized."""
    connector = _get_connector(kv_cache_manager_with_disk)
    engine = connector._engine
    assert engine.storage_manager is not None
    backends = list(engine.storage_manager.storage_backends.keys())

    assert "LocalCPUBackend" in backends, (
        f"Expected LocalCPUBackend in {backends}"
    )
    assert "LocalDiskBackend" in backends, (
        f"Expected LocalDiskBackend in {backends}"
    )


def test_disk_tier_storage(
    kv_cache_manager_with_disk: _TPPagedKVCacheManager,
) -> None:
    """Verify blocks are written to disk tier by querying it directly."""
    import time

    tp_mgr = kv_cache_manager_with_disk
    connector = _get_connector(tp_mgr)

    device_tensor = tp_mgr.device_tensors[0]
    pattern = np.arange(device_tensor.to_numpy().size, dtype=np.float32)
    pattern = pattern.reshape(device_tensor.shape)
    device_tensor.inplace_copy_from(Buffer.from_numpy(pattern))

    block_ids = list(range(8))
    block_hashes = [21000 + i for i in block_ids]
    connector.save(block_ids, block_hashes)
    connector.flush()

    # Wait for async disk write to complete
    time.sleep(0.5)

    disk_hits = query_backend(
        connector,
        block_hashes,
        block_size=INTEGRATION_PAGE_SIZE,
        backend_name="LocalDiskBackend",
    )
    assert disk_hits == len(block_hashes), (
        f"Expected {len(block_hashes)} blocks in disk, found {disk_hits}"
    )

    ctx = make_dummy_context()
    tokens = connector.lookup(ctx, block_hashes)
    assert tokens == len(block_ids) * INTEGRATION_PAGE_SIZE

    target_block_ids = list(range(32, 40))
    loaded_hashes = connector.load(ctx, target_block_ids, [])
    assert loaded_hashes == block_hashes


def test_tiered_storage_roundtrip(
    kv_cache_manager_with_disk: _TPPagedKVCacheManager,
) -> None:
    """Save -> clear device -> load -> verify data integrity."""
    tp_mgr = kv_cache_manager_with_disk
    connector = _get_connector(tp_mgr)

    test_config = KVCacheTestConfig(
        num_blocks=64,
        kv_dim=2,
        num_layers=2,
        page_size=INTEGRATION_PAGE_SIZE,
        num_kv_heads=4,
        head_dim=64,
    )

    device_tensor = tp_mgr.device_tensors[0]
    original_pattern = fill_paged_cache(device_tensor, test_config)

    block_ids = list(range(8))
    block_hashes = [30000 + i for i in block_ids]
    connector.save(block_ids, block_hashes)
    connector.flush()

    # Clear the device tensor to verify data is actually loaded
    device_tensor.inplace_copy_from(
        Buffer.zeros(
            shape=device_tensor.shape,
            dtype=device_tensor.dtype,
            device=device_tensor.device,
        )
    )
    cleared_data = device_tensor.to_numpy()
    assert np.all(cleared_data == 0), "Device tensor should be cleared"

    ctx = make_dummy_context()
    tokens = connector.lookup(ctx, block_hashes)
    assert tokens == len(block_ids) * INTEGRATION_PAGE_SIZE

    loaded_hashes = connector.load(ctx, block_ids, [])
    assert loaded_hashes == block_hashes

    loaded_data = device_tensor.to_numpy()
    for block_id in block_ids:
        np.testing.assert_array_almost_equal(
            loaded_data[block_id],
            original_pattern[block_id],
            decimal=5,
            err_msg=f"Data mismatch for block {block_id}",
        )


def test_tiered_storage_pattern_verification(
    kv_cache_manager_with_disk: _TPPagedKVCacheManager,
) -> None:
    """Precise bit-level save/load verification."""
    tp_mgr = kv_cache_manager_with_disk
    connector = _get_connector(tp_mgr)

    test_config = KVCacheTestConfig(
        num_blocks=64,
        kv_dim=2,
        num_layers=2,
        page_size=INTEGRATION_PAGE_SIZE,
        num_kv_heads=4,
        head_dim=64,
    )

    device_tensor = tp_mgr.device_tensors[0]
    original_pattern = fill_paged_cache(device_tensor, test_config)

    block_ids = list(range(8))
    block_hashes = [40000 + i for i in block_ids]
    connector.save(block_ids, block_hashes)
    connector.flush()

    # Clear device tensor
    device_tensor.inplace_copy_from(
        Buffer.zeros(
            shape=device_tensor.shape,
            dtype=device_tensor.dtype,
            device=device_tensor.device,
        )
    )

    ctx = make_dummy_context()
    connector.lookup(ctx, block_hashes)
    connector.load(ctx, block_ids, [])

    loaded_data = device_tensor.to_numpy()
    for block_id in block_ids:
        expected = original_pattern[block_id]
        actual = loaded_data[block_id]

        if not np.allclose(expected, actual, rtol=1e-5, atol=1e-5):
            diff = np.abs(expected - actual)
            max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
            raise AssertionError(
                f"Block {block_id} data mismatch. "
                f"Max diff at {max_diff_idx}: "
                f"expected {expected[max_diff_idx]}, "
                f"got {actual[max_diff_idx]}"
            )


def test_multiple_requests_with_tiered_storage(
    kv_cache_manager_with_disk: _TPPagedKVCacheManager,
) -> None:
    """Multiple batches can be stored and accessed independently."""
    tp_mgr = kv_cache_manager_with_disk
    connector = _get_connector(tp_mgr)

    device_tensor = tp_mgr.device_tensors[0]
    pattern = np.arange(device_tensor.to_numpy().size, dtype=np.float32)
    pattern = pattern.reshape(device_tensor.shape)
    device_tensor.inplace_copy_from(Buffer.from_numpy(pattern))

    # First batch
    batch1_ids = list(range(8))
    batch1_hashes = [50000 + i for i in batch1_ids]
    connector.save(batch1_ids, batch1_hashes)
    connector.flush()

    # Second batch
    batch2_ids = list(range(8, 16))
    batch2_hashes = [60000 + i for i in range(8)]
    connector.save(batch2_ids, batch2_hashes)
    connector.flush()

    ctx1 = make_dummy_context()
    tokens1 = connector.lookup(ctx1, batch1_hashes)
    assert tokens1 == len(batch1_hashes) * INTEGRATION_PAGE_SIZE

    loaded1 = connector.load(ctx1, list(range(32, 40)), [])
    assert loaded1 == batch1_hashes

    ctx2 = create_text_context(np.array([4, 5, 6], dtype=np.int64))
    tokens2 = connector.lookup(ctx2, batch2_hashes)
    assert tokens2 == len(batch2_hashes) * INTEGRATION_PAGE_SIZE

    loaded2 = connector.load(ctx2, list(range(40, 48)), [])
    assert loaded2 == batch2_hashes
