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

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.interfaces import RequestID
from max.kv_cache import PagedKVCacheManager
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams
from test_common.context_utils import create_text_context


def _make_kv_manager(
    *,
    num_devices: int = 1,
    page_size: int = 128,
    total_num_pages: int = 8,
    n_kv_heads: int = 1,
    head_dim: int = 16,
    num_layers: int = 10,
    max_batch_size: int = 128,
    devices: list[DeviceRef] | None = None,
    **extra_kv_params: object,
) -> PagedKVCacheManager:
    devices = [DeviceRef.CPU() for _ in range(num_devices)]
    params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        page_size=page_size,
        devices=devices or [DeviceRef.CPU()],
        **extra_kv_params,  # type: ignore[arg-type]
    )
    return PagedKVCacheManager(
        params=params,
        session=InferenceSession(devices=[CPU()]),
        total_num_pages=total_num_pages,
        max_batch_size=max_batch_size,
    )


@pytest.mark.asyncio
async def test_step() -> None:
    kv_manager = _make_kv_manager(n_kv_heads=8, head_dim=128)

    prompt_lens = [3, 4, 7]
    batch = []
    for i in range(3):
        context = create_text_context(np.empty(prompt_lens[i]))
        kv_manager.claim(context.request_id, replica_idx=0)
        batch.append(context)

    # Assert that each cache_length is initialized appropriately as 0
    for ctx in batch:
        assert ctx.tokens.processed_length == 0

    # Update these values a few times
    for j in range(3):
        for ctx in batch:
            kv_manager.alloc(ctx, replica_idx=0, num_steps=1)
        kv_manager.runtime_inputs([batch])
        for ctx in batch:
            ctx.update(42)
        kv_manager.step([batch])

        for i, ctx in enumerate(batch):
            assert ctx.tokens.processed_length == prompt_lens[i] * (j + 1)

        for i, ctx in enumerate(batch):
            orig_start_idx = ctx.tokens.processed_length
            for _ in range(prompt_lens[i] - 1):
                ctx.update(42)

            ctx.tokens.rewind_processing(
                ctx.tokens.processed_length - orig_start_idx
            )


@pytest.mark.asyncio
async def test_claim_and_release() -> None:
    # Initialize llama like params
    # claim and release are both cache_type independent,
    # so we can test with the KVCacheType.CONTINUOUS default
    kv_manager = _make_kv_manager(n_kv_heads=8, head_dim=128)
    # TODO: This test should not access internal _replica
    replica = kv_manager._replica[0]

    contexts = []
    prompt_lens = [2, 3, 4, 5, 6]
    for i in range(5):
        context = create_text_context(np.empty(prompt_lens[i]))
        kv_manager.claim(context.request_id, replica_idx=0)
        contexts.append(context)

    # Claim 5 ids
    assert len(contexts) == 5
    assert len(replica.claimed_requests) == 5

    # Claim another 3 ids
    contexts_2 = []
    prompt_lens_2 = [7, 8, 9]
    for i in range(3):
        context = create_text_context(np.empty(prompt_lens_2[i]))
        kv_manager.claim(context.request_id, replica_idx=0)
        contexts_2.append(context)

    assert len(replica.claimed_requests) == 5 + 3

    # Release id that has not been claimed
    with pytest.raises(ValueError):
        kv_manager.release(RequestID("fake-request-id"), replica_idx=0)

    # Release all ids
    for i, context in enumerate(contexts + contexts_2):
        kv_manager.release(context.request_id, replica_idx=0)
        assert len(replica.claimed_requests) == 5 + 3 - i - 1


@pytest.mark.asyncio
async def test_fetch_paged() -> None:
    kv_manager = _make_kv_manager()

    # Claim 5 items
    contexts = []
    for _ in range(5):
        context = create_text_context(np.empty(1))
        kv_manager.claim(context.request_id, replica_idx=0)
        contexts.append(context)

    # Fetch 3 of the 5 contexts created above
    for ctx in contexts[:3]:
        kv_manager.alloc(ctx, replica_idx=0, num_steps=1)
    _ = kv_manager.runtime_inputs([contexts[:3]]).inputs[0]


@pytest.mark.asyncio
async def test_reserve_claims_and_releases() -> None:
    kv_manager = _make_kv_manager()
    contexts = [
        create_text_context(np.zeros(1, dtype=np.int64)) for _ in range(2)
    ]

    with kv_manager.reserve([contexts], num_steps=1):
        for context in contexts:
            assert kv_manager.contains(context.request_id, replica_idx=0)

    for context in contexts:
        assert not kv_manager.contains(context.request_id, replica_idx=0)


@pytest.mark.asyncio
async def test_fetch_paged_lookup_table_tracks_required_page_capacity() -> None:
    kv_manager = _make_kv_manager()

    short_context = create_text_context(np.zeros(1, dtype=np.int64))
    kv_manager.claim(short_context.request_id, replica_idx=0)

    kv_manager.alloc(short_context, replica_idx=0, num_steps=1)
    first_inputs = kv_manager.runtime_inputs([[short_context]]).inputs[0]
    assert tuple(first_inputs.lookup_table.shape) == (1, 1)

    long_context = create_text_context(np.zeros(256, dtype=np.int64))
    kv_manager.claim(long_context.request_id, replica_idx=0)

    kv_manager.alloc(long_context, replica_idx=0, num_steps=1)
    second_inputs = kv_manager.runtime_inputs([[long_context]]).inputs[0]
    assert tuple(second_inputs.lookup_table.shape) == (1, 2)


@pytest.mark.asyncio
async def test_runtime_inputs_lookup_table_uses_explicit_max_cache_length() -> (
    None
):
    total_num_pages = 8
    kv_manager = _make_kv_manager(total_num_pages=total_num_pages)

    context = create_text_context(np.zeros(1, dtype=np.int64))
    kv_manager.claim(context.request_id, replica_idx=0)
    kv_manager.alloc(context, replica_idx=0, num_steps=1)

    runtime_inputs = kv_manager.runtime_inputs([[context]]).inputs[0]
    assert tuple(runtime_inputs.lookup_table.shape) == (1, 1)

    explicit_inputs = kv_manager.runtime_inputs(
        [[context]],
        max_cache_length=1024,
        num_steps=1,
    ).inputs[0]
    assert tuple(explicit_inputs.lookup_table.shape) == (1, total_num_pages)


@pytest.mark.asyncio
async def test_mla_runtime_inputs_handles_empty_replica_batch() -> None:
    """MLA runtime inputs should handle empty per-replica batches.

    This exercises data-parallel serving where one replica has work and another
    replica is idle in the same scheduler iteration.
    """
    kv_manager = _make_kv_manager(
        num_devices=2,
        data_parallel_degree=2,
        is_mla=True,
        num_q_heads=8,
    )

    context = create_text_context(np.zeros(1, dtype=np.int64))
    kv_manager.claim(context.request_id, replica_idx=0)
    kv_manager.alloc(context, replica_idx=0, num_steps=1)

    runtime_inputs = kv_manager.runtime_inputs([[context], []], num_steps=1)
    assert len(runtime_inputs.inputs) == 2
    assert runtime_inputs.inputs[0].attention_dispatch_metadata is not None
    assert runtime_inputs.inputs[1].attention_dispatch_metadata is not None


@pytest.mark.asyncio
async def test_alloc_num_speculative_steps_allocates_extra_blocks() -> None:
    """alloc with num_speculative_steps reserves blocks for spec tokens."""
    page_size = 4
    kv_manager = _make_kv_manager(page_size=page_size, total_num_pages=16)

    ctx = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    kv_manager.claim(ctx.request_id, replica_idx=0)

    # Without speculative steps: 3 tokens + 1 step - 1 = 3 → 1 block
    kv_manager.alloc(ctx, replica_idx=0, num_steps=1)
    blocks_base = len(
        kv_manager._replica[0].block_manager.req_to_blocks[ctx.request_id]
    )

    kv_manager.release(ctx.request_id, replica_idx=0)

    # With speculative steps: 3 + 0 draft + 4 spec + 1 - 1 = 7 → 2 blocks
    ctx2 = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    kv_manager.claim(ctx2.request_id, replica_idx=0)
    kv_manager.alloc(ctx2, replica_idx=0, num_steps=1, num_speculative_steps=4)
    blocks_spec = len(
        kv_manager._replica[0].block_manager.req_to_blocks[ctx2.request_id]
    )

    assert blocks_spec > blocks_base


@pytest.mark.asyncio
async def test_alloc_with_saved_draft_tokens_reserves_more_blocks() -> None:
    """alloc accounts for saved_draft_tokens in the context."""
    page_size = 4
    kv_manager = _make_kv_manager(page_size=page_size, total_num_pages=16)

    ctx = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    ctx.spec_decoding_state.saved_draft_tokens = [10, 20, 30]
    kv_manager.claim(ctx.request_id, replica_idx=0)
    # seq_len = 3 tokens + 3 draft + 4 spec + 1 - 1 = 10 → 3 blocks
    kv_manager.alloc(ctx, replica_idx=0, num_steps=1, num_speculative_steps=4)
    blocks = len(
        kv_manager._replica[0].block_manager.req_to_blocks[ctx.request_id]
    )
    assert blocks == 3


@pytest.mark.asyncio
async def test_runtime_inputs_with_num_speculative_steps() -> None:
    """runtime_inputs passes through num_speculative_steps without error."""
    page_size = 4
    kv_manager = _make_kv_manager(page_size=page_size, total_num_pages=16)

    ctx = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    kv_manager.claim(ctx.request_id, replica_idx=0)
    kv_manager.alloc(ctx, replica_idx=0, num_steps=1, num_speculative_steps=4)

    inputs = kv_manager.runtime_inputs(
        [[ctx]], num_steps=1, num_speculative_steps=4
    )
    assert len(inputs.inputs) == 1


@pytest.mark.asyncio
async def test_runtime_inputs_raises_when_spec_blocks_missing() -> None:
    """runtime_inputs raises if alloc was called without enough spec space."""
    page_size = 4
    kv_manager = _make_kv_manager(page_size=page_size, total_num_pages=16)

    ctx = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    kv_manager.claim(ctx.request_id, replica_idx=0)
    # Allocate without speculative steps
    kv_manager.alloc(ctx, replica_idx=0, num_steps=1, num_speculative_steps=0)

    # Request runtime_inputs with speculative steps → should fail
    with pytest.raises(ValueError, match="does not have sufficient blocks"):
        kv_manager.runtime_inputs([[ctx]], num_steps=1, num_speculative_steps=4)


def _make_multi_kv_manager(
    *,
    page_size: int = 128,
    total_num_pages: int = 8,
    max_batch_size: int = 128,
    enable_prefix_caching: bool = False,
) -> PagedKVCacheManager:
    """Creates a multi-cache manager with two caches (primary + secondary)."""
    devices = [DeviceRef.CPU()]
    primary = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=8,
        head_dim=128,
        num_layers=10,
        page_size=page_size,
        devices=devices,
        enable_prefix_caching=enable_prefix_caching,
    )
    secondary = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=1,
        head_dim=64,
        num_layers=10,
        page_size=page_size,
        devices=devices,
        enable_prefix_caching=enable_prefix_caching,
    )
    multi_params = MultiKVCacheParams.from_params(primary, secondary)
    return PagedKVCacheManager(
        params=multi_params,
        session=InferenceSession(devices=[CPU()]),
        total_num_pages=total_num_pages,
        max_batch_size=max_batch_size,
    )


@pytest.mark.asyncio
async def test_multi_cache_alloc_skip_tokens_is_safe() -> None:
    """skip_tokens=True is safe for multi-cache because there is one BlockManager.

    Previously, models with multiple KV caches used separate
    PagedKVCacheManagers. Each manager had its own BlockManager, so
    alloc(skip_tokens=True) on one manager would mutate
    ctx.tokens.processed_length before the second manager allocated,
    causing the second alloc to see stale token state.

    With a single multi-cache PagedKVCacheManager, there is one shared
    BlockManager. alloc() is called once and internally handles all
    caches, so skip_tokens=True cannot cause a state mismatch.
    """
    page_size = 128
    kv_manager = _make_multi_kv_manager(
        page_size=page_size,
        total_num_pages=16,
        enable_prefix_caching=True,
    )
    assert kv_manager.num_caches == 2

    # --- First request: populate the prefix cache ---
    ctx1 = create_text_context(
        np.arange(page_size + 1, dtype=np.int64), max_length=2048
    )
    kv_manager.claim(ctx1.request_id, replica_idx=0)
    kv_manager.alloc(ctx1, replica_idx=0, num_steps=1)
    assert ctx1.tokens.processed_length == 0

    # Simulate a full decode step so the first page gets committed.
    kv_manager.runtime_inputs([[ctx1]])
    ctx1.update(42)
    kv_manager.step([[ctx1]])

    kv_manager.release(ctx1.request_id, replica_idx=0)

    # --- Second request: same prefix, should get a prefix cache hit ---
    ctx2 = create_text_context(
        np.arange(page_size + 1, dtype=np.int64), max_length=2048
    )
    kv_manager.claim(ctx2.request_id, replica_idx=0)
    assert ctx2.tokens.processed_length == 0

    # alloc applies prefix-cache skip internally. With the old separate-manager
    # approach, this would have required skip_tokens=False to avoid corrupting
    # state between managers. A single multi-cache manager has one BlockManager,
    # so the skip is applied once safely.
    kv_manager.alloc(ctx2, replica_idx=0, num_steps=1)
    assert ctx2.tokens.processed_length == page_size

    # Verify the context is in a consistent state for runtime_inputs.
    kv_manager.runtime_inputs([[ctx2]])


@pytest.mark.asyncio
async def test_multi_cache_runtime_inputs_combined() -> None:
    """runtime_inputs returns combined inputs for all caches."""
    kv_manager = _make_multi_kv_manager(total_num_pages=16)
    assert kv_manager.num_caches == 2

    ctx = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    kv_manager.claim(ctx.request_id, replica_idx=0)
    kv_manager.alloc(ctx, replica_idx=0, num_steps=1)

    inputs = kv_manager.runtime_inputs([[ctx]])

    # With 1 device and 2 caches, combined inputs should have 2 entries.
    assert len(inputs.inputs) == 2

    # Both entries share the same cache_lengths and lookup_table buffers.
    assert inputs.inputs[0].cache_lengths is inputs.inputs[1].cache_lengths
    assert inputs.inputs[0].lookup_table is inputs.inputs[1].lookup_table

    # But they have different block buffers (different caches).
    assert inputs.inputs[0].blocks is not inputs.inputs[1].blocks


@pytest.mark.asyncio
async def test_multi_cache_lifecycle() -> None:
    """claim/alloc/step/release work across all caches with one call each."""
    kv_manager = _make_multi_kv_manager(total_num_pages=16)

    ctx = create_text_context(np.array([1, 2, 3], dtype=np.int64))

    # Single claim covers all caches.
    kv_manager.claim(ctx.request_id, replica_idx=0)
    assert kv_manager.contains(ctx.request_id, replica_idx=0)

    # Single alloc covers all caches.
    kv_manager.alloc(ctx, replica_idx=0, num_steps=1)
    kv_manager.runtime_inputs([[ctx]])
    ctx.update(42)

    # Single step covers all caches.
    kv_manager.step([[ctx]])

    # Single release covers all caches.
    kv_manager.release(ctx.request_id, replica_idx=0)
    assert not kv_manager.contains(ctx.request_id, replica_idx=0)
