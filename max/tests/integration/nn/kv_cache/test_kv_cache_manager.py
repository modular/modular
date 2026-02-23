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
from max.nn.legacy.kv_cache import KVCacheParams
from test_common.context_utils import create_text_context


@pytest.mark.asyncio
async def test_step() -> None:
    # Initialize llama like params
    # Step is cache_type agnostic, so we can test with contiguous
    device = CPU()
    num_layers = 10
    params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=8,
        head_dim=128,
        num_layers=num_layers,
        cache_strategy="paged",
        page_size=128,
        devices=[DeviceRef.from_device(device)],
    )

    kv_manager = PagedKVCacheManager(
        params=params,
        session=InferenceSession(devices=[device]),
        total_num_pages=8,
        max_batch_size=128,
    )

    # Create three text contexts and externally claim each using their request_id
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
    device = CPU()
    params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=8,
        head_dim=128,
        num_layers=10,
        cache_strategy="paged",
        page_size=128,
        devices=[DeviceRef.CPU()],
    )

    kv_manager = PagedKVCacheManager(
        params=params,
        session=InferenceSession(devices=[device]),
        total_num_pages=8,
        max_batch_size=128,
    )
    # This test requires PagedKVCacheManager to access internal _replica_managers
    assert isinstance(kv_manager, PagedKVCacheManager), (
        "test_claim_and_release requires PagedKVCacheManager"
    )
    replica_manager = kv_manager._replica_managers[0]

    contexts = []
    prompt_lens = [2, 3, 4, 5, 6]
    for i in range(5):
        context = create_text_context(np.empty(prompt_lens[i]))
        kv_manager.claim(context.request_id, replica_idx=0)
        contexts.append(context)

    # Claim 5 ids
    assert len(contexts) == 5
    assert len(replica_manager._claimed_requests) == 5

    # Claim another 3 ids
    contexts_2 = []
    prompt_lens_2 = [7, 8, 9]
    for i in range(3):
        context = create_text_context(np.empty(prompt_lens_2[i]))
        kv_manager.claim(context.request_id, replica_idx=0)
        contexts_2.append(context)

    assert len(replica_manager._claimed_requests) == 5 + 3

    # Release id that has not been claimed
    with pytest.raises(ValueError):
        kv_manager.release(RequestID("fake-request-id"), replica_idx=0)

    # Release all ids
    for i, context in enumerate(contexts + contexts_2):
        kv_manager.release(context.request_id, replica_idx=0)
        assert len(replica_manager._claimed_requests) == 5 + 3 - i - 1


@pytest.mark.asyncio
async def test_fetch_paged() -> None:
    # Initialize llama like params
    device = CPU()
    params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=1,
        head_dim=16,
        num_layers=10,
        cache_strategy="paged",
        page_size=128,
        devices=[DeviceRef.CPU()],
    )

    kv_manager = PagedKVCacheManager(
        params=params,
        session=InferenceSession(devices=[device]),
        total_num_pages=8,
        max_batch_size=128,
    )

    # Claim 5 items
    contexts = []
    for _ in range(5):
        context = create_text_context(np.empty(1))
        kv_manager.claim(context.request_id, replica_idx=0)
        contexts.append(context)

    # Fetch 3 of the 5 contexts created above
    for ctx in contexts[:3]:
        kv_manager.alloc(ctx, replica_idx=0, num_steps=1)
    kv_collection = kv_manager.runtime_inputs([contexts[:3]])[0]

    assert kv_collection is not None


@pytest.mark.asyncio
async def test_reserve_claims_and_releases() -> None:
    device = CPU()
    params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=1,
        head_dim=16,
        num_layers=10,
        cache_strategy="paged",
        page_size=128,
        devices=[DeviceRef.CPU()],
    )

    kv_manager = PagedKVCacheManager(
        params=params,
        session=InferenceSession(devices=[device]),
        total_num_pages=8,
        max_batch_size=128,
    )
    contexts = [
        create_text_context(np.zeros(1, dtype=np.int64)) for _ in range(2)
    ]

    with kv_manager.reserve(contexts, replica_idx=0, num_steps=1):
        for context in contexts:
            assert kv_manager.contains(context.request_id, replica_idx=0)

    for context in contexts:
        assert not kv_manager.contains(context.request_id, replica_idx=0)


@pytest.mark.asyncio
async def test_fetch_paged_lookup_table_tracks_required_page_capacity() -> None:
    device = CPU()
    total_num_pages = 8
    params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=1,
        head_dim=16,
        num_layers=10,
        cache_strategy="paged",
        page_size=128,
        devices=[DeviceRef.CPU()],
    )

    kv_manager = PagedKVCacheManager(
        params=params,
        session=InferenceSession(devices=[device]),
        total_num_pages=total_num_pages,
        max_batch_size=128,
    )

    short_context = create_text_context(np.zeros(1, dtype=np.int64))
    kv_manager.claim(short_context.request_id, replica_idx=0)

    kv_manager.alloc(short_context, replica_idx=0, num_steps=1)
    first_inputs = kv_manager.runtime_inputs([[short_context]])[0]
    assert tuple(first_inputs.lookup_table.shape) == (1, 1)

    long_context = create_text_context(np.zeros(256, dtype=np.int64))
    kv_manager.claim(long_context.request_id, replica_idx=0)

    kv_manager.alloc(long_context, replica_idx=0, num_steps=1)
    second_inputs = kv_manager.runtime_inputs([[long_context]])[0]
    assert tuple(second_inputs.lookup_table.shape) == (1, 2)
