# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.nn.kv_cache import (
    KVCacheParams,
    KVCacheStrategy,
    load_kv_manager,
)
from test_common.context_utils import create_text_context


@pytest.mark.asyncio
async def test_step() -> None:
    # Initialize llama like params
    # Step is cache_type agnostic, so we can test with contiguous
    device = CPU()
    params = KVCacheParams(dtype=DType.float32, n_kv_heads=8, head_dim=128)

    kv_manager = load_kv_manager(
        params=params,
        max_batch_size=16,
        max_seq_len=100,
        num_layers=10,
        devices=[device],
        session=InferenceSession(devices=[device]),
    )

    # Create three text contexts and externally claim each using their request_id
    prompt_lens = [3, 4, 7]
    batch = []
    for i in range(3):
        seq_id = list(kv_manager.available)[0]
        context = create_text_context(seq_id, np.empty(prompt_lens[i]))
        kv_manager.external_claim(context.request_id)
        batch.append(context)

    # Assert that each cache_length is initialized appropriately as 0
    for ctx in batch:
        assert ctx.start_idx == 0

    # Update these values a few times
    for j in range(3):
        kv_manager.fetch(batch)
        for ctx in batch:
            ctx.update(42)
        kv_manager.step(batch)

        for i, ctx in enumerate(batch):
            assert ctx.start_idx == prompt_lens[i] * (j + 1)

        for i, ctx in enumerate(batch):
            orig_start_idx = ctx.start_idx
            for _ in range(prompt_lens[i] - 1):
                ctx.update(42)
            ctx.set_token_indices(start_idx=orig_start_idx)


@pytest.mark.asyncio
async def test_claim_and_release() -> None:
    # Initialize llama like params
    # claim and release are both cache_type independent,
    # so we can test with the KVCacheType.CONTINUOUS default
    device = CPU()
    params = KVCacheParams(dtype=DType.float32, n_kv_heads=8, head_dim=128)

    kv_manager = load_kv_manager(
        params=params,
        max_batch_size=16,
        max_seq_len=100,
        num_layers=10,
        devices=[device],
        session=InferenceSession(devices=[device]),
    )

    contexts = []
    prompt_lens = [2, 3, 4, 5, 6]
    for i in range(5):
        seq_id = list(kv_manager.available)[0]
        context = create_text_context(seq_id, np.empty(prompt_lens[i]))
        kv_manager.external_claim(context.request_id)
        contexts.append(context)

    # Claim 5 ids
    outstanding = 11
    assert len(contexts) == 5
    assert len(kv_manager.available) == outstanding

    # Claim another 3 ids
    contexts_2 = []
    prompt_lens_2 = [7, 8, 9]
    for i in range(3):
        seq_id = list(kv_manager.available)[0]
        context = create_text_context(seq_id, np.empty(prompt_lens_2[i]))
        kv_manager.external_claim(context.request_id)
        contexts_2.append(context)

    outstanding -= 3
    assert len(kv_manager.available) == outstanding

    # Release id that has not been claimed
    with pytest.raises(ValueError):
        kv_manager.release("fake-request-id")

    # Release all ids
    for i, context in enumerate(contexts + contexts_2):
        kv_manager.release(context.request_id)
        assert len(kv_manager.available) == outstanding + i + 1


@pytest.mark.asyncio
async def test_fetch_continuous() -> None:
    # Initialize llama like params
    device = CPU()
    params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=1,
        head_dim=16,
        cache_strategy=KVCacheStrategy.CONTINUOUS,
    )

    kv_manager = load_kv_manager(
        params=params,
        max_batch_size=16,
        max_seq_len=100,
        num_layers=10,
        devices=[device],
        session=InferenceSession(devices=[device]),
    )

    # Raise on fetch when nothing has been claimed
    with pytest.raises(ValueError):
        bogus_seq_id = 100
        kv_collection = kv_manager.fetch(
            [create_text_context(bogus_seq_id, np.empty(1))]
        )[0]

    # Claim 5 items
    seq_ids = []
    contexts = []
    for _ in range(5):
        seq_id = list(kv_manager.available)[0]
        context = create_text_context(seq_id, np.empty(1))
        kv_manager.external_claim(context.request_id)
        seq_ids.append(seq_id)
        contexts.append(context)

    # Fetch 3 of the 5 ids
    # Fetch 3 of the 5 contexts created above
    kv_collection = kv_manager.fetch(contexts[:3])[0]

    assert kv_collection is not None
