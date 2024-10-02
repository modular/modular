# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
import asyncio
from nn.kv_cache import KVCacheParams, ContinuousBatchingKVCacheManager
from max.engine import InferenceSession
from max.driver import CPU
from max.dtype import DType


def test_step(session: InferenceSession):
    asyncio.run(_test_step(session))


async def _test_step(session: InferenceSession):
    # Initialize llama like params
    device = CPU()
    params = KVCacheParams(
        dtype=DType.float32, n_kv_heads=8, head_dim=128, device=device
    )

    kv_manager = ContinuousBatchingKVCacheManager(
        params=params,
        max_cache_size=16,
        max_seq_len=100,
        num_layers=10,
        session=session,
        device=device,
    )

    # Claim three items
    seq_ids = await kv_manager.claim(n=3)

    # Assert that each cache_length is initialized appropriately as 0
    for seq_id in seq_ids:
        assert kv_manager.cache_lengths[seq_id] == 0

    # Update these values a few times
    values = [3, 4, 7]
    for j in range(3):
        valid_lengths = {}
        for i, seq_id in enumerate(seq_ids):
            valid_lengths[seq_id] = values[i]

        kv_manager.step(valid_lengths)

        for i, seq_id in enumerate(seq_ids):
            assert kv_manager.cache_lengths[seq_id] == values[i] * (j + 1)


def test_claim_and_release(session: InferenceSession):
    asyncio.run(_test_claim_and_release(session))


async def _test_claim_and_release(session: InferenceSession):
    # Initialize llama like params
    device = CPU()
    params = KVCacheParams(
        dtype=DType.float32, n_kv_heads=8, head_dim=128, device=device
    )

    kv_manager = ContinuousBatchingKVCacheManager(
        params=params,
        max_cache_size=16,
        max_seq_len=100,
        num_layers=10,
        session=session,
        device=device,
    )

    # Claim 5 ids
    outstanding = 11
    seq_ids = await kv_manager.claim(n=5)
    assert len(seq_ids) == 5
    assert kv_manager.slots_remaining == outstanding

    # Claim another 3 ids
    seq_ids_2 = await kv_manager.claim(n=3)
    assert len(seq_ids_2) == 3
    outstanding -= 3
    assert kv_manager.slots_remaining == outstanding

    # Release id that has not been claimed
    with pytest.raises(ValueError):
        await kv_manager.release(seq_id=25)

    # Release all ids
    for i, id in enumerate(seq_ids + seq_ids_2):
        await kv_manager.release(seq_id=id)
        assert kv_manager.slots_remaining == outstanding + i + 1


@pytest.mark.skip("MSDK-1079: Implement Fetch")
def test_fetch(session: InferenceSession):
    asyncio.run(_test_fetch(session))


async def _test_fetch(session: InferenceSession):
    # Initialize llama like params
    device = CPU()
    params = KVCacheParams(
        dtype=DType.float32, n_kv_heads=8, head_dim=128, device=device
    )

    kv_manager = ContinuousBatchingKVCacheManager(
        params=params,
        max_cache_size=16,
        max_seq_len=100,
        num_layers=10,
        session=session,
        device=device,
    )

    # Raise on fetch when nothing has been claimed
    with pytest.raises(ValueError):
        kv_collection = kv_manager.fetch(seq_ids=[0])

    # Claim 5 items
    seq_ids = await kv_manager.claim(n=5)

    # Fetch 3 of the 5 ids
    kv_collection = kv_manager.fetch(seq_ids[:3])
    assert kv_collection is not None
