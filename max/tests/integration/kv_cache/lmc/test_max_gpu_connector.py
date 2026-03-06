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

"""GPU kernel tests for MAXGPUConnector offload/onload operations.

These tests verify the correctness of the Mojo kernels used for transferring
KV cache data between MAX's paged format and LMCache's external format.

Test categories:

1. **Round-trip integrity**: Verify data survives offload -> onload cycles.
2. **Slot mapping**: Contiguous, sparse, and reversed slot mappings.
3. **Partial ranges**: Non-zero start indices, sub-ranges of slot mappings.
4. **Scaling**: Realistic model dimensions.
5. **Edge cases**: Empty ranges, block boundaries, last slot.
"""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_count
from max.engine import InferenceSession
from max.kv_cache.connectors.lmcache_connector import MAXGPUConnector

from .conftest import (
    MODEL_CONFIG,
    SMALL_CONFIG,
    KVCacheTestConfig,
    cleanup_gpu_connector,
    create_memory_obj,
    fill_paged_cache,
)
from .conftest import (
    create_max_gpu_connector as _create_max_gpu_connector,
)

# Cache of (connector, paged_cache) by config, so tests sharing a config
# reuse the same compiled Mojo graphs instead of compiling new ones.
_connector_cache: dict[KVCacheTestConfig, tuple[MAXGPUConnector, Buffer]] = {}


def get_connector(
    config: KVCacheTestConfig,
    device: Accelerator,
    session: InferenceSession,
) -> tuple[MAXGPUConnector, Buffer]:
    """Get or create a (connector, paged_cache) pair for the given config."""
    if config not in _connector_cache:
        connector, paged_cache, _, _ = _create_max_gpu_connector(
            config, device=device, session=session
        )
        _connector_cache[config] = (connector, paged_cache)
    return _connector_cache[config]


@pytest.fixture(scope="module")
def gpu_resources() -> Generator[
    tuple[Accelerator, InferenceSession], None, None
]:
    """Shared GPU device and session for all MAXGPUConnector tests."""
    if accelerator_count() == 0:
        pytest.skip("No GPU available")
    device = Accelerator()
    session = InferenceSession(devices=[device])
    yield device, session
    for connector, _ in _connector_cache.values():
        cleanup_gpu_connector(connector)
    _connector_cache.clear()


def test_round_trip(
    gpu_resources: tuple[Accelerator, InferenceSession],
) -> None:
    """Test offload -> clear -> onload round-trip with deterministic data.

    Uses contiguous slots spanning multiple blocks to cover basic gather,
    scatter, and cross-block-boundary behavior in a single test.
    """
    device, session = gpu_resources
    config = SMALL_CONFIG
    connector, paged_cache = get_connector(config, device, session)
    original_data = fill_paged_cache(paged_cache, config)

    # Span 3 blocks (12 tokens with page_size=4)
    num_tokens = config.page_size * 3
    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device="cuda")

    gathered_obj = create_memory_obj(
        shape=(
            config.kv_dim,
            config.num_layers,
            num_tokens,
            config.hidden_dim,
        ),
        dtype=torch.float32,
    )
    connector.from_gpu(
        gathered_obj, start=0, end=num_tokens, slot_mapping=slot_mapping
    )

    assert gathered_obj.tensor is not None
    assert not np.allclose(gathered_obj.tensor.numpy(), 0)

    # Clear paged cache and scatter back
    paged_cache.inplace_copy_from(
        Buffer.from_numpy(np.zeros_like(original_data))
    )
    connector.to_gpu(
        gathered_obj, start=0, end=num_tokens, slot_mapping=slot_mapping
    )

    restored_data = paged_cache.to_numpy()
    for token_idx in range(num_tokens):
        block_id = token_idx // config.page_size
        offset = token_idx % config.page_size

        for kv_idx in range(config.kv_dim):
            for layer_idx in range(config.num_layers):
                original = original_data[
                    block_id, kv_idx, layer_idx, offset, :, :
                ]
                restored = restored_data[
                    block_id, kv_idx, layer_idx, offset, :, :
                ]
                assert np.allclose(original, restored, rtol=1e-5), (
                    f"Round-trip failed at token {token_idx}"
                )


def test_round_trip_random_data(
    gpu_resources: tuple[Accelerator, InferenceSession],
) -> None:
    """Test round-trip with random data to catch dtype/precision issues."""
    device, session = gpu_resources
    config = SMALL_CONFIG
    connector, paged_cache = get_connector(config, device, session)

    np.random.seed(42)
    original_data = np.random.randn(*config.paged_cache_shape).astype(
        np.float32
    )
    paged_cache.inplace_copy_from(Buffer.from_numpy(original_data))

    num_tokens = config.page_size * 3
    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device="cuda")

    gathered_obj = create_memory_obj(
        shape=(
            config.kv_dim,
            config.num_layers,
            num_tokens,
            config.hidden_dim,
        ),
        dtype=torch.float32,
    )
    connector.from_gpu(
        gathered_obj, start=0, end=num_tokens, slot_mapping=slot_mapping
    )

    paged_cache.inplace_copy_from(
        Buffer.from_numpy(np.zeros_like(original_data))
    )
    connector.to_gpu(
        gathered_obj, start=0, end=num_tokens, slot_mapping=slot_mapping
    )

    restored_data = paged_cache.to_numpy()
    for block_id in range(3):
        for offset in range(config.page_size):
            for kv_idx in range(config.kv_dim):
                for layer_idx in range(config.num_layers):
                    original = original_data[
                        block_id, kv_idx, layer_idx, offset, :, :
                    ]
                    restored = restored_data[
                        block_id, kv_idx, layer_idx, offset, :, :
                    ]
                    assert np.allclose(original, restored, rtol=1e-5), (
                        f"Round-trip failed at block={block_id}, offset={offset}"
                    )


def test_sparse_slot_mapping(
    gpu_resources: tuple[Accelerator, InferenceSession],
) -> None:
    """Test gather and scatter with non-contiguous slot mappings.

    Covers sparse (every-other) and reversed slot patterns, verifying
    that non-mapped slots remain zeroed after scatter.
    """
    device, session = gpu_resources
    config = SMALL_CONFIG
    connector, paged_cache = get_connector(config, device, session)
    original_data = fill_paged_cache(paged_cache, config)

    # Sparse mapping: every other slot
    sparse_slots = [0, 2, 4, 6]
    num_tokens = len(sparse_slots)
    slot_mapping = torch.tensor(sparse_slots, dtype=torch.long, device="cuda")

    gathered_obj = create_memory_obj(
        shape=(
            config.kv_dim,
            config.num_layers,
            num_tokens,
            config.hidden_dim,
        ),
        dtype=torch.float32,
    )
    connector.from_gpu(
        gathered_obj, start=0, end=num_tokens, slot_mapping=slot_mapping
    )

    assert gathered_obj.tensor is not None
    gathered_data = gathered_obj.tensor.numpy()

    for token_idx, slot in enumerate(sparse_slots):
        block_id = slot // config.page_size
        offset = slot % config.page_size

        for kv_idx in range(config.kv_dim):
            for layer_idx in range(config.num_layers):
                expected = (
                    block_id * 1000 + kv_idx * 100 + layer_idx * 10 + offset
                )
                actual = gathered_data[kv_idx, layer_idx, token_idx, 0]
                assert np.isclose(expected, actual), (
                    f"Sparse gather failed at token {token_idx}, slot {slot}"
                )

    # Scatter to sparse slots on a zeroed cache and verify non-mapped stay zero
    paged_cache.inplace_copy_from(
        Buffer.from_numpy(np.zeros_like(original_data))
    )
    connector.to_gpu(
        gathered_obj, start=0, end=num_tokens, slot_mapping=slot_mapping
    )

    restored_data = paged_cache.to_numpy()
    for slot in range(config.num_blocks * config.page_size):
        block_id = slot // config.page_size
        offset = slot % config.page_size

        if slot in sparse_slots:
            token_idx = sparse_slots.index(slot)
            src_block = slot // config.page_size
            src_offset = slot % config.page_size

            for kv_idx in range(config.kv_dim):
                for layer_idx in range(config.num_layers):
                    expected = (
                        src_block * 1000
                        + kv_idx * 100
                        + layer_idx * 10
                        + src_offset
                    )
                    actual = restored_data[
                        block_id, kv_idx, layer_idx, offset, 0, 0
                    ]
                    assert np.isclose(expected, actual), (
                        f"Sparse scatter mismatch at slot {slot}"
                    )
        else:
            actual = restored_data[block_id, :, :, offset, :, :]
            assert np.allclose(actual, 0), (
                f"Non-mapped slot {slot} should be zero"
            )


def test_nonzero_start_range(
    gpu_resources: tuple[Accelerator, InferenceSession],
) -> None:
    """Test gather/scatter with non-zero start/end indices.

    Gathers a sub-range of the slot mapping, then scatters to a different
    offset, verifying the start/end kernel parameters work correctly.
    """
    device, session = gpu_resources
    config = SMALL_CONFIG
    connector, paged_cache = get_connector(config, device, session)
    original_data = fill_paged_cache(paged_cache, config)

    # Full slot mapping, but only gather from indices 2-6
    total_slots = 20
    full_slot_mapping = torch.arange(
        total_slots, dtype=torch.long, device="cuda"
    )
    gather_start, gather_end = 2, 10
    num_tokens = gather_end - gather_start

    gathered_obj = create_memory_obj(
        shape=(
            config.kv_dim,
            config.num_layers,
            num_tokens,
            config.hidden_dim,
        ),
        dtype=torch.float32,
    )
    connector.from_gpu(
        gathered_obj,
        start=gather_start,
        end=gather_end,
        slot_mapping=full_slot_mapping,
    )

    assert gathered_obj.tensor is not None
    gathered_data = gathered_obj.tensor.numpy()

    # Verify gather read from the correct slots
    for out_idx in range(num_tokens):
        slot = gather_start + out_idx
        block_id = slot // config.page_size
        offset = slot % config.page_size

        for kv_idx in range(config.kv_dim):
            for layer_idx in range(config.num_layers):
                expected = (
                    block_id * 1000 + kv_idx * 100 + layer_idx * 10 + offset
                )
                actual = gathered_data[kv_idx, layer_idx, out_idx, 0]
                assert np.isclose(expected, actual), (
                    f"Nonzero-start gather failed: out_idx={out_idx}, slot={slot}"
                )

    # Scatter to a different range
    paged_cache.inplace_copy_from(
        Buffer.from_numpy(np.zeros_like(original_data))
    )
    scatter_start = 10
    scatter_slot_mapping = torch.arange(
        scatter_start, scatter_start + 20, dtype=torch.long, device="cuda"
    )
    connector.to_gpu(
        gathered_obj,
        start=0,
        end=num_tokens,
        slot_mapping=scatter_slot_mapping,
    )

    restored_data = paged_cache.to_numpy()
    for i in range(num_tokens):
        src_slot = gather_start + i
        dst_slot = scatter_start + i

        src_block = src_slot // config.page_size
        src_offset = src_slot % config.page_size
        dst_block = dst_slot // config.page_size
        dst_offset = dst_slot % config.page_size

        for kv_idx in range(config.kv_dim):
            for layer_idx in range(config.num_layers):
                expected = (
                    src_block * 1000
                    + kv_idx * 100
                    + layer_idx * 10
                    + src_offset
                )
                actual = restored_data[
                    dst_block, kv_idx, layer_idx, dst_offset, 0, 0
                ]
                assert np.isclose(expected, actual), (
                    f"Nonzero-start scatter failed: "
                    f"src_slot={src_slot}, dst_slot={dst_slot}"
                )


def test_model_dimensions(
    gpu_resources: tuple[Accelerator, InferenceSession],
) -> None:
    """Test round-trip with realistic model-scale dimensions."""
    device, session = gpu_resources
    config = MODEL_CONFIG
    num_blocks_to_test = 2
    connector, paged_cache = get_connector(config, device, session)
    original_data = fill_paged_cache(paged_cache, config)

    num_tokens = config.page_size * num_blocks_to_test
    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device="cuda")

    gathered_obj = create_memory_obj(
        shape=(
            config.kv_dim,
            config.num_layers,
            num_tokens,
            config.hidden_dim,
        ),
        dtype=torch.float32,
    )
    connector.from_gpu(
        gathered_obj, start=0, end=num_tokens, slot_mapping=slot_mapping
    )

    paged_cache.inplace_copy_from(
        Buffer.from_numpy(np.zeros_like(original_data))
    )
    connector.to_gpu(
        gathered_obj, start=0, end=num_tokens, slot_mapping=slot_mapping
    )

    restored_data = paged_cache.to_numpy()
    mismatches = 0
    max_error = 0.0
    for block_id in range(num_blocks_to_test):
        for offset in range(config.page_size):
            for kv_idx in range(config.kv_dim):
                for layer_idx in range(config.num_layers):
                    original = original_data[
                        block_id, kv_idx, layer_idx, offset, :, :
                    ]
                    restored = restored_data[
                        block_id, kv_idx, layer_idx, offset, :, :
                    ]
                    error = np.abs(original - restored).max()
                    max_error = max(max_error, error)
                    if not np.allclose(original, restored, rtol=1e-5):
                        mismatches += 1

    assert mismatches == 0, (
        f"Model dimension test failed: {mismatches} mismatches, "
        f"max_error={max_error}"
    )


def test_edge_cases(
    gpu_resources: tuple[Accelerator, InferenceSession],
) -> None:
    """Test edge cases: empty range, block boundaries, last slot."""
    device, session = gpu_resources
    config = SMALL_CONFIG
    connector, paged_cache = get_connector(config, device, session)
    fill_paged_cache(paged_cache, config)

    # Empty range (start == end) should not error
    slot_mapping = torch.arange(8, dtype=torch.long, device="cuda")
    empty_obj = create_memory_obj(
        shape=(config.kv_dim, config.num_layers, 0, config.hidden_dim),
        dtype=torch.float32,
    )
    connector.from_gpu(empty_obj, start=0, end=0, slot_mapping=slot_mapping)
    connector.to_gpu(empty_obj, start=0, end=0, slot_mapping=slot_mapping)

    # Block boundary slots: last slot of block 0, first slot of block 1
    for slot in [config.page_size - 1, config.page_size]:
        sm = torch.tensor([slot], dtype=torch.long, device="cuda")
        obj = create_memory_obj(
            shape=(config.kv_dim, config.num_layers, 1, config.hidden_dim),
            dtype=torch.float32,
        )
        connector.from_gpu(obj, start=0, end=1, slot_mapping=sm)

        assert obj.tensor is not None
        block_id = slot // config.page_size
        offset = slot % config.page_size

        for kv_idx in range(config.kv_dim):
            for layer_idx in range(config.num_layers):
                expected = (
                    block_id * 1000 + kv_idx * 100 + layer_idx * 10 + offset
                )
                actual = obj.tensor[kv_idx, layer_idx, 0, 0].item()
                assert np.isclose(expected, actual), (
                    f"Boundary test failed at slot {slot}"
                )

    # Last slot in entire cache
    last_slot = config.num_blocks * config.page_size - 1
    sm = torch.tensor([last_slot], dtype=torch.long, device="cuda")
    obj = create_memory_obj(
        shape=(config.kv_dim, config.num_layers, 1, config.hidden_dim),
        dtype=torch.float32,
    )
    connector.from_gpu(obj, start=0, end=1, slot_mapping=sm)

    assert obj.tensor is not None
    block_id = last_slot // config.page_size
    offset = last_slot % config.page_size

    for kv_idx in range(config.kv_dim):
        for layer_idx in range(config.num_layers):
            expected = block_id * 1000 + kv_idx * 100 + layer_idx * 10 + offset
            actual = obj.tensor[kv_idx, layer_idx, 0, 0].item()
            assert np.isclose(expected, actual), (
                f"Last slot test failed: expected={expected}, actual={actual}"
            )
