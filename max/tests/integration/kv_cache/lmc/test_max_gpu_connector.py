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

1. **Basic data transfer**: Verify offload/onload correctly read/write paged cache.
2. **Round-trip integrity**: Verify data survives offload -> onload cycles.
3. **Slot mapping**: Contiguous, sparse, and reversed slot mappings.
4. **Partial ranges**: Non-zero start indices, sub-ranges of slot mappings.
5. **Scaling**: Small and realistic model dimensions (8B-scale).
6. **Edge cases**: Empty ranges, block boundaries, last slot.
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
    LARGE_MODEL_CONFIG,
    MEDIUM_CONFIG,
    SMALL_CONFIG,
    SMALL_MODEL_CONFIG,
    KVCacheTestConfig,
    cleanup_gpu_connector,
    create_memory_obj,
    fill_paged_cache,
)
from .conftest import (
    create_max_gpu_connector as _create_max_gpu_connector,
)

# Inline configs used by multiple tests — defined once so the connector
# cache (keyed by config) can reuse compiled Mojo graphs.
_8BLOCK_CONFIG = KVCacheTestConfig(
    num_blocks=8,
    kv_dim=2,
    num_layers=2,
    page_size=4,
    num_kv_heads=2,
    head_dim=4,
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


@pytest.mark.parametrize(
    "num_tokens,description",
    [
        pytest.param(1, "single_token", id="single"),
        pytest.param(6, "contiguous_tokens", id="contiguous"),
    ],
)
def test_gather(
    gpu_resources: tuple[Accelerator, InferenceSession],
    num_tokens: int,
    description: str,
) -> None:
    """Test gathering tokens from paged cache."""
    device, session = gpu_resources
    config = SMALL_CONFIG
    connector, paged_cache = get_connector(config, device, session)
    fill_paged_cache(paged_cache, config)

    # For single token, gather from slot 5 (block 1, offset 1)
    # For multiple tokens, gather contiguous slots starting at 0
    if num_tokens == 1:
        slot_mapping = torch.tensor([5], dtype=torch.long, device="cuda")
    else:
        slot_mapping = torch.arange(num_tokens, dtype=torch.long, device="cuda")

    memory_obj = create_memory_obj(
        shape=(
            config.kv_dim,
            config.num_layers,
            num_tokens,
            config.hidden_dim,
        ),
        dtype=torch.float32,
    )

    connector.from_gpu(
        memory_obj, start=0, end=num_tokens, slot_mapping=slot_mapping
    )

    assert memory_obj.tensor is not None
    output = memory_obj.tensor.numpy()

    mismatches = 0
    for token_idx in range(num_tokens):
        slot = slot_mapping[token_idx].item()
        block_id = slot // config.page_size
        offset = slot % config.page_size

        for kv_idx in range(config.kv_dim):
            for layer_idx in range(config.num_layers):
                expected_value = (
                    block_id * 1000 + kv_idx * 100 + layer_idx * 10 + offset
                )
                actual = output[kv_idx, layer_idx, token_idx, 0]
                if not np.isclose(expected_value, actual):
                    mismatches += 1

    assert mismatches == 0, (
        f"Gather {description} failed: {mismatches} mismatches"
    )


def test_gather_cross_block_boundary(
    gpu_resources: tuple[Accelerator, InferenceSession],
) -> None:
    """Test gathering tokens that span multiple blocks."""
    device, session = gpu_resources
    config = SMALL_CONFIG
    connector, paged_cache = get_connector(config, device, session)
    fill_paged_cache(paged_cache, config)

    # Gather slots 2-9 (spans blocks 0, 1, 2)
    num_tokens = 8
    slot_mapping = torch.arange(
        2, 2 + num_tokens, dtype=torch.long, device="cuda"
    )
    memory_obj = create_memory_obj(
        shape=(
            config.kv_dim,
            config.num_layers,
            num_tokens,
            config.hidden_dim,
        ),
        dtype=torch.float32,
    )

    connector.from_gpu(
        memory_obj, start=0, end=num_tokens, slot_mapping=slot_mapping
    )

    assert memory_obj.tensor is not None
    output = memory_obj.tensor.numpy()

    for token_idx in range(num_tokens):
        slot = 2 + token_idx
        block_id = slot // config.page_size
        offset = slot % config.page_size

        for kv_idx in range(config.kv_dim):
            for layer_idx in range(config.num_layers):
                expected_value = (
                    block_id * 1000 + kv_idx * 100 + layer_idx * 10 + offset
                )
                actual = output[kv_idx, layer_idx, token_idx, 0]
                assert np.isclose(expected_value, actual), (
                    f"Cross-block gather failed at token {token_idx}, slot {slot}"
                )


@pytest.mark.parametrize(
    "num_tokens,description",
    [
        pytest.param(1, "single_token", id="single"),
        pytest.param(6, "contiguous_tokens", id="contiguous"),
    ],
)
def test_scatter(
    gpu_resources: tuple[Accelerator, InferenceSession],
    num_tokens: int,
    description: str,
) -> None:
    """Test scattering tokens to paged cache."""
    device, session = gpu_resources
    config = SMALL_CONFIG
    connector, paged_cache = get_connector(config, device, session)

    # Zero the cache since it may have data from prior tests sharing this config
    paged_cache.inplace_copy_from(
        Buffer.from_numpy(np.zeros(config.paged_cache_shape, dtype=np.float32))
    )

    memory_obj = create_memory_obj(
        shape=(
            config.kv_dim,
            config.num_layers,
            num_tokens,
            config.hidden_dim,
        ),
        dtype=torch.float32,
    )
    assert memory_obj.tensor is not None

    if num_tokens == 1:
        # Fill with known value and scatter to slot 7
        memory_obj.tensor.fill_(42.0)
        slot_mapping = torch.tensor([7], dtype=torch.long, device="cuda")
    else:
        # Fill with token-indexed values
        for token_idx in range(num_tokens):
            for kv_idx in range(config.kv_dim):
                for layer_idx in range(config.num_layers):
                    value = token_idx * 1000 + kv_idx * 100 + layer_idx * 10
                    memory_obj.tensor[kv_idx, layer_idx, token_idx, :] = value
        slot_mapping = torch.arange(num_tokens, dtype=torch.long, device="cuda")

    connector.to_gpu(
        memory_obj, start=0, end=num_tokens, slot_mapping=slot_mapping
    )

    paged_cache_np = paged_cache.to_numpy()

    mismatches = 0
    for token_idx in range(num_tokens):
        slot = slot_mapping[token_idx].item()
        block_id = slot // config.page_size
        offset = slot % config.page_size

        if num_tokens == 1:
            expected = 42.0
        else:
            expected = token_idx * 1000

        actual = paged_cache_np[block_id, 0, 0, offset, 0, 0]
        if not np.isclose(expected, actual):
            mismatches += 1

    assert mismatches == 0, (
        f"Scatter {description} failed: {mismatches} mismatches"
    )

    # Verify non-mapped slots remain zero (for single token case)
    if num_tokens == 1:
        actual = paged_cache_np[0, 0, 0, 0, 0, 0]
        assert np.isclose(0.0, actual), "Non-mapped slots should remain zero"


def test_round_trip_basic(
    gpu_resources: tuple[Accelerator, InferenceSession],
) -> None:
    """Test basic round-trip: gather, clear, scatter, verify."""
    device, session = gpu_resources
    config = SMALL_CONFIG
    connector, paged_cache = get_connector(config, device, session)
    original_data = fill_paged_cache(paged_cache, config)

    num_tokens = config.page_size * 2
    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device="cuda")

    # Gather
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
    assert not np.allclose(gathered_obj.tensor.numpy(), 0), (
        "Gathered data should not be all zeros"
    )

    # Clear paged cache
    paged_cache.inplace_copy_from(
        Buffer.from_numpy(np.zeros_like(original_data))
    )

    # Scatter back
    connector.to_gpu(
        gathered_obj, start=0, end=num_tokens, slot_mapping=slot_mapping
    )

    # Verify
    restored_data = paged_cache.to_numpy()
    mismatches = 0
    max_error = 0.0

    for token_idx in range(num_tokens):
        slot = token_idx
        block_id = slot // config.page_size
        offset = slot % config.page_size

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
        f"Round-trip failed: {mismatches} mismatches, max_error={max_error}"
    )


@pytest.mark.parametrize("seed", [42, 123, 456])
def test_round_trip_with_random_data(
    gpu_resources: tuple[Accelerator, InferenceSession],
    seed: int,
) -> None:
    """Test round-trip with random data patterns."""
    device, session = gpu_resources
    config = MEDIUM_CONFIG
    connector, paged_cache = get_connector(config, device, session)

    np.random.seed(seed)
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

    # Check first 3 blocks
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
                        f"Round-trip failed at seed={seed}, block={block_id}"
                    )


def test_sparse_slot_mapping(
    gpu_resources: tuple[Accelerator, InferenceSession],
) -> None:
    """Test with slots that skip positions (e.g., 0, 2, 4, 6)."""
    device, session = gpu_resources
    config = _8BLOCK_CONFIG
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

    # Verify gathered data matches the sparse slots
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

    # Scatter to different sparse slots
    paged_cache.inplace_copy_from(
        Buffer.from_numpy(np.zeros_like(original_data))
    )

    connector.to_gpu(
        gathered_obj, start=0, end=num_tokens, slot_mapping=slot_mapping
    )

    restored_data = paged_cache.to_numpy()

    # Verify only mapped slots have data
    for slot in range(config.num_blocks * config.page_size):
        block_id = slot // config.page_size
        offset = slot % config.page_size

        if slot in sparse_slots:
            token_idx = sparse_slots.index(slot)
            src_slot = sparse_slots[token_idx]
            src_block = src_slot // config.page_size
            src_offset = src_slot % config.page_size

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
            # Non-mapped slots should be zero
            actual = restored_data[block_id, :, :, offset, :, :]
            assert np.allclose(actual, 0), (
                f"Non-mapped slot {slot} should be zero"
            )


def test_reverse_slot_mapping(
    gpu_resources: tuple[Accelerator, InferenceSession],
) -> None:
    """Test with reversed slot order."""
    device, session = gpu_resources
    config = SMALL_CONFIG
    connector, paged_cache = get_connector(config, device, session)
    fill_paged_cache(paged_cache, config)

    # Reversed mapping: 7, 6, 5, 4
    reversed_slots = [7, 6, 5, 4]
    num_tokens = len(reversed_slots)
    slot_mapping = torch.tensor(reversed_slots, dtype=torch.long, device="cuda")

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

    # output[i] should have data from slot reversed_slots[i]
    for token_idx, slot in enumerate(reversed_slots):
        block_id = slot // config.page_size
        offset = slot % config.page_size

        for kv_idx in range(config.kv_dim):
            for layer_idx in range(config.num_layers):
                expected = (
                    block_id * 1000 + kv_idx * 100 + layer_idx * 10 + offset
                )
                actual = gathered_data[kv_idx, layer_idx, token_idx, 0]
                assert np.isclose(expected, actual), (
                    f"Reversed gather failed at token {token_idx}, slot {slot}"
                )


def test_gather_with_nonzero_start(
    gpu_resources: tuple[Accelerator, InferenceSession],
) -> None:
    """Test gathering from middle of slot_mapping.

    Key behavior:
    - slot_mapping contains slots for all tokens
    - start/end specify which portion of slot_mapping to use
    - output tensor is sized for (end - start) tokens
    - data is written starting at position 0 in the output
    """
    device, session = gpu_resources
    config = SMALL_CONFIG
    connector, paged_cache = get_connector(config, device, session)
    fill_paged_cache(paged_cache, config)

    # Full slot mapping for 8 tokens (slots 0-7)
    total_slots = 8
    full_slot_mapping = torch.arange(
        total_slots, dtype=torch.long, device="cuda"
    )

    # Gather from index 2-6 (4 tokens, using slots 2,3,4,5)
    start, end = 2, 6
    num_tokens = end - start

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
        gathered_obj, start=start, end=end, slot_mapping=full_slot_mapping
    )

    assert gathered_obj.tensor is not None
    gathered_data = gathered_obj.tensor.numpy()

    # output[i] should have data from slot (start + i)
    for out_idx in range(num_tokens):
        slot = start + out_idx  # slots 2, 3, 4, 5
        block_id = slot // config.page_size
        offset = slot % config.page_size

        for kv_idx in range(config.kv_dim):
            for layer_idx in range(config.num_layers):
                expected = (
                    block_id * 1000 + kv_idx * 100 + layer_idx * 10 + offset
                )
                actual = gathered_data[kv_idx, layer_idx, out_idx, 0]
                assert np.isclose(expected, actual), (
                    f"Partial gather failed: out_idx={out_idx}, slot={slot}, "
                    f"expected={expected}, actual={actual}"
                )


def test_scatter_with_nonzero_start(
    gpu_resources: tuple[Accelerator, InferenceSession],
) -> None:
    """Test scattering to middle of slot_mapping."""
    device, session = gpu_resources
    config = _8BLOCK_CONFIG
    connector, paged_cache = get_connector(config, device, session)

    # Zero the cache since it may have data from prior tests sharing this config
    paged_cache.inplace_copy_from(
        Buffer.from_numpy(np.zeros(config.paged_cache_shape, dtype=np.float32))
    )

    # Create memory obj with pattern based on token index
    num_tokens = 4
    memory_obj = create_memory_obj(
        shape=(
            config.kv_dim,
            config.num_layers,
            num_tokens,
            config.hidden_dim,
        ),
        dtype=torch.float32,
    )
    assert memory_obj.tensor is not None
    for token_idx in range(num_tokens):
        memory_obj.tensor[:, :, token_idx, :] = token_idx * 100 + 1

    # Full slot mapping
    total_slots = 16
    full_slot_mapping = torch.arange(
        total_slots, dtype=torch.long, device="cuda"
    )

    # Scatter using indices 4-8 (slots 4,5,6,7)
    start, end = 4, 8
    connector.to_gpu(
        memory_obj, start=start, end=end, slot_mapping=full_slot_mapping
    )

    paged_cache_np = paged_cache.to_numpy()

    # Verify: slots 4-7 should have data from tokens 0-3
    for token_idx in range(num_tokens):
        slot = start + token_idx
        block_id = slot // config.page_size
        offset = slot % config.page_size

        expected = token_idx * 100 + 1
        actual = paged_cache_np[block_id, 0, 0, offset, 0, 0]
        assert np.isclose(expected, actual), (
            f"Partial scatter failed: token_idx={token_idx}, slot={slot}, "
            f"expected={expected}, actual={actual}"
        )

    # Slots 0-3 should still be zero
    for slot in range(4):
        block_id = slot // config.page_size
        offset = slot % config.page_size
        actual = paged_cache_np[block_id, :, :, offset, :, :]
        assert np.allclose(actual, 0), f"Slot {slot} should be zero"


def test_partial_range_round_trip(
    gpu_resources: tuple[Accelerator, InferenceSession],
) -> None:
    """Test round-trip with partial ranges."""
    device, session = gpu_resources
    config = _8BLOCK_CONFIG
    connector, paged_cache = get_connector(config, device, session)
    original_data = fill_paged_cache(paged_cache, config)

    # Full slot mapping
    total_slots = 20
    full_slot_mapping = torch.arange(
        total_slots, dtype=torch.long, device="cuda"
    )

    # Gather from range 6-14 (8 tokens)
    gather_start, gather_end = 6, 14
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

    # Clear the cache
    paged_cache.inplace_copy_from(
        Buffer.from_numpy(np.zeros_like(original_data))
    )

    # Scatter to a different range: 10-18
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

    # Verify: data from slots 6-13 should now be at slots 10-17
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
                    f"Partial range round-trip failed: "
                    f"src_slot={src_slot}, dst_slot={dst_slot}, "
                    f"expected={expected}, actual={actual}"
                )


@pytest.mark.parametrize(
    "config,num_blocks_to_test",
    [
        pytest.param(SMALL_MODEL_CONFIG, 4, id="small_model"),
        pytest.param(LARGE_MODEL_CONFIG, 2, id="8b_model"),
    ],
)
def test_model_dimensions(
    gpu_resources: tuple[Accelerator, InferenceSession],
    config: KVCacheTestConfig,
    num_blocks_to_test: int,
) -> None:
    """Test with various model-scale dimensions."""
    device, session = gpu_resources
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

    # Verify the tested blocks
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


def test_empty_range(
    gpu_resources: tuple[Accelerator, InferenceSession],
) -> None:
    """Test with start == end (no tokens)."""
    device, session = gpu_resources
    config = SMALL_CONFIG
    connector, paged_cache = get_connector(config, device, session)
    fill_paged_cache(paged_cache, config)

    slot_mapping = torch.arange(8, dtype=torch.long, device="cuda")
    memory_obj = create_memory_obj(
        shape=(config.kv_dim, config.num_layers, 0, config.hidden_dim),
        dtype=torch.float32,
    )

    # Should handle gracefully without error
    connector.from_gpu(memory_obj, start=0, end=0, slot_mapping=slot_mapping)
    connector.to_gpu(memory_obj, start=0, end=0, slot_mapping=slot_mapping)


def test_single_element_at_boundary(
    gpu_resources: tuple[Accelerator, InferenceSession],
) -> None:
    """Test single element at block boundaries."""
    device, session = gpu_resources
    config = SMALL_CONFIG
    connector, paged_cache = get_connector(config, device, session)
    fill_paged_cache(paged_cache, config)

    # Test at block boundaries: last slot of block 0, first slot of block 1
    boundary_slots = [config.page_size - 1, config.page_size]

    for slot in boundary_slots:
        slot_mapping = torch.tensor([slot], dtype=torch.long, device="cuda")
        memory_obj = create_memory_obj(
            shape=(config.kv_dim, config.num_layers, 1, config.hidden_dim),
            dtype=torch.float32,
        )

        connector.from_gpu(
            memory_obj, start=0, end=1, slot_mapping=slot_mapping
        )

        assert memory_obj.tensor is not None
        block_id = slot // config.page_size
        offset = slot % config.page_size

        for kv_idx in range(config.kv_dim):
            for layer_idx in range(config.num_layers):
                expected = (
                    block_id * 1000 + kv_idx * 100 + layer_idx * 10 + offset
                )
                actual = memory_obj.tensor[kv_idx, layer_idx, 0, 0].item()
                assert np.isclose(expected, actual), (
                    f"Boundary test failed at slot {slot}"
                )


def test_last_slot_in_cache(
    gpu_resources: tuple[Accelerator, InferenceSession],
) -> None:
    """Test gathering from the last slot in the cache."""
    device, session = gpu_resources
    config = SMALL_CONFIG
    connector, paged_cache = get_connector(config, device, session)
    fill_paged_cache(paged_cache, config)

    last_slot = config.num_blocks * config.page_size - 1
    slot_mapping = torch.tensor([last_slot], dtype=torch.long, device="cuda")
    memory_obj = create_memory_obj(
        shape=(config.kv_dim, config.num_layers, 1, config.hidden_dim),
        dtype=torch.float32,
    )

    connector.from_gpu(memory_obj, start=0, end=1, slot_mapping=slot_mapping)

    assert memory_obj.tensor is not None
    block_id = last_slot // config.page_size
    offset = last_slot % config.page_size

    for kv_idx in range(config.kv_dim):
        for layer_idx in range(config.num_layers):
            expected = block_id * 1000 + kv_idx * 100 + layer_idx * 10 + offset
            actual = memory_obj.tensor[kv_idx, layer_idx, 0, 0].item()
            assert np.isclose(expected, actual), (
                f"Last slot test failed: expected={expected}, actual={actual}"
            )
