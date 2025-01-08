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
from max.pipelines.kv_cache import (
    KVCacheParams,
    KVCacheStrategy,
    load_kv_manager,
)

FAKE_TOKEN = 999


@pytest.mark.asyncio
async def test_step():
    # Initialize llama like params
    # Step is cache_type agnostic, so we can test with contiguous
    device = CPU()
    params = KVCacheParams(dtype=DType.float32, n_kv_heads=8, head_dim=128)

    kv_manager = load_kv_manager(
        params=params,
        max_cache_batch_size=16,
        max_seq_len=100,
        num_layers=10,
        devices=[device],
        session=InferenceSession(devices=[device]),
    )

    # Claim three items
    seq_ids = kv_manager.claim(n=3)

    # Assert that each cache_length is initialized appropriately as 0
    for seq_id in seq_ids:
        assert kv_manager.cache_lengths[seq_id] == 0

    # Update these values a few times
    prompt_lens = [3, 4, 7]
    for j in range(3):
        seq_ids_and_prompts = {}
        for i, seq_id in enumerate(seq_ids):
            seq_ids_and_prompts[seq_id] = np.array(
                [FAKE_TOKEN] * prompt_lens[i]
            )

        kv_manager.fetch(seq_ids_and_prompts)
        seq_ids_and_new_tokens = {
            seq_id: np.array([FAKE_TOKEN]) for seq_id in seq_ids
        }
        kv_manager.step(seq_ids_and_new_tokens)

        for i, seq_id in enumerate(seq_ids):
            assert kv_manager.cache_lengths[seq_id] == prompt_lens[i] * (j + 1)


@pytest.mark.asyncio
async def test_claim_and_release():
    # Initialize llama like params
    # claim and release are both cache_type independent,
    # so we can test with the KVCacheType.CONTINUOUS default
    device = CPU()
    params = KVCacheParams(dtype=DType.float32, n_kv_heads=8, head_dim=128)

    kv_manager = load_kv_manager(
        params=params,
        max_cache_batch_size=16,
        max_seq_len=100,
        num_layers=10,
        devices=[device],
        session=InferenceSession(devices=[device]),
    )

    # Claim 5 ids
    outstanding = 11
    seq_ids = kv_manager.claim(n=5)
    assert len(seq_ids) == 5
    assert len(kv_manager.slots_remaining) == outstanding

    # Claim another 3 ids
    seq_ids_2 = kv_manager.claim(n=3)
    assert len(seq_ids_2) == 3
    outstanding -= 3
    assert len(kv_manager.slots_remaining) == outstanding

    # Release id that has not been claimed
    with pytest.raises(ValueError):
        kv_manager.release(seq_id=25)

    # Release all ids
    for i, id in enumerate(seq_ids + seq_ids_2):
        kv_manager.release(seq_id=id)
        assert len(kv_manager.slots_remaining) == outstanding + i + 1


@pytest.mark.asyncio
async def test_fetch_continuous():
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
        max_cache_batch_size=16,
        max_seq_len=100,
        num_layers=10,
        devices=[device],
        session=InferenceSession(devices=[device]),
    )

    # Raise on fetch when nothing has been claimed
    with pytest.raises(ValueError):
        bogus_seq_id = 100
        kv_collection = kv_manager.fetch(
            {bogus_seq_id: np.array([FAKE_TOKEN])}
        )[0]

    # Claim 5 items
    seq_ids = kv_manager.claim(n=5)

    # Fetch 3 of the 5 ids
    kv_collection = kv_manager.fetch(
        {s: np.array([FAKE_TOKEN]) for s in seq_ids[:3]}
    )[0]
    assert kv_collection is not None
