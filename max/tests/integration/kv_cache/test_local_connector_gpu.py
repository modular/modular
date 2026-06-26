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

import pytest
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheBuffer, KVConnectorType, MHAKVCacheParams
from max.pipelines.kv_cache.connectors.local_connector import LocalConnector
from max.pipelines.kv_cache.kv_connector import to_block_hash_bytes


# Test-fixture helper for the bytes-only connector boundary: tests pre-date
# Option A and used int hashes directly; ``_hs`` wraps a variadic sequence of
# ints into the canonical 8-byte signed-BE encoding.
def _hs(*ns: int) -> list[bytes]:
    return [to_block_hash_bytes(n) for n in ns]


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
    kv_params = MHAKVCacheParams(
        dtype=DType.float32,
        num_layers=num_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        enable_prefix_caching=True,
        kv_connector=KVConnectorType.local,
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

    return LocalConnector(
        replica_kv_memory=[kv_buffer.to_memory()],
        total_num_host_blocks=num_host_blocks,
    )


def test_connector_name() -> None:
    """Verify LocalConnector has correct name."""
    connector = create_local_connector()
    assert connector.name == "LocalConnector"


def test_num_host_blocks() -> None:
    """Verify num_host_blocks returns the configured value."""
    num_host = 48
    connector = create_local_connector(num_host_blocks=num_host)
    assert connector.num_host_blocks == num_host


def test_num_used_host_blocks_initially_zero() -> None:
    """Verify no host blocks are used initially."""
    connector = create_local_connector()
    assert connector.num_used_host_blocks == 0


def test_offload_transfers_blocks_to_host() -> None:
    """Verify offload() transfers blocks to host cache."""
    connector = create_local_connector()

    connector.offload([0, 1], _hs(100, 200))

    assert connector.num_used_host_blocks == 2


def test_duplicate_hash_not_saved_twice() -> None:
    """Verify blocks with same hash are deduplicated."""
    connector = create_local_connector()

    connector.offload([0], _hs(100))
    connector.offload([1], _hs(100))

    assert connector.num_used_host_blocks == 1


def test_load_returns_zero_for_empty_cache() -> None:
    """Verify load returns 0 when cache is empty."""
    connector = create_local_connector(page_size=16)

    loaded = connector.load([0, 1, 2], _hs(100, 200, 300))

    assert loaded == 0


def test_load_finds_cached_blocks() -> None:
    """Verify load returns correct number of loaded blocks."""
    connector = create_local_connector(page_size=16)

    connector.offload([0, 1, 2], _hs(100, 200, 300))

    loaded = connector.load([3, 4, 5], _hs(100, 200, 300))

    assert loaded == 3


def test_load_stops_at_first_miss() -> None:
    """Verify load returns contiguous prefix only."""
    connector = create_local_connector(page_size=16)

    connector.offload([0], _hs(100))
    connector.offload([2], _hs(300))

    loaded = connector.load([3, 4], _hs(100, 200))

    assert loaded == 1


def test_load_full_round_trip() -> None:
    """Verify full prefix cache hit: save -> load round-trip."""
    connector = create_local_connector(page_size=16)

    connector.offload([0, 1, 2], _hs(100, 200, 300))
    assert connector.num_used_host_blocks == 3

    loaded = connector.load([10, 11, 12], _hs(100, 200, 300))
    assert loaded == 3

    assert connector.num_used_host_blocks == 3


def test_load_partial_hit() -> None:
    """Verify partial prefix cache hit returns only matching prefix."""
    connector = create_local_connector(page_size=16)

    connector.offload([0, 1], _hs(100, 200))

    loaded = connector.load([10, 11, 12], _hs(100, 200, 300))
    assert loaded == 2


def test_load_miss_at_start() -> None:
    """Verify cache miss at start of sequence returns nothing."""
    connector = create_local_connector(page_size=16)

    connector.offload([1, 2], _hs(200, 300))

    loaded = connector.load([10, 11, 12], _hs(100, 200, 300))
    assert loaded == 0


def test_reset_prefix_cache_clears_host_cache() -> None:
    """Verify reset_prefix_cache clears all cached blocks."""
    connector = create_local_connector()

    connector.offload([0, 1, 2], _hs(100, 200, 300))
    assert connector.num_used_host_blocks == 3

    connector.reset_prefix_cache()

    assert connector.num_used_host_blocks == 0


def test_shutdown_completes_cleanly() -> None:
    """Verify shutdown waits for transfers and completes cleanly."""
    connector = create_local_connector()

    connector.offload([0, 1], _hs(100, 200))

    connector.shutdown()


def test_repeated_load_does_not_leak() -> None:
    """Verify N rounds of load don't accumulate leaked blocks."""
    connector = create_local_connector(num_host_blocks=32)

    connector.offload([0], _hs(100))

    free_baseline = connector._host_block_pool.free_block_queue.num_free_blocks

    for _i in range(5):
        loaded = connector.load([10], _hs(100))
        assert loaded == 1

    free_after_cycles = (
        connector._host_block_pool.free_block_queue.num_free_blocks
    )
    assert free_after_cycles == free_baseline, (
        f"Free block count should be stable after repeated load "
        f"cycles: expected {free_baseline}, got {free_after_cycles}"
    )
