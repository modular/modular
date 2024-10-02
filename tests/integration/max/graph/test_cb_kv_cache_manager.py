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


@pytest.mark.skip("MSDK-1078: Implement Claim and Release")
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
        max_batch_size=16,
        max_seq_len=100,
        num_layers=10,
        session=session,
        device=device,
    )

    # Claim 5 ids
    outstanding = 11
    seq_ids = await kv_manager.claim(batch_size=5)
    assert len(seq_ids) == 5
    assert kv_manager.slots_remaining == outstanding

    # Claim another 3 ids
    seq_ids_2 = await kv_manager.claim(batch_size=3)
    assert len(seq_ids_2) == 3
    outstanding -= 3
    assert kv_manager.slots_remaining == outstanding

    # Release id that has not been claimed
    with pytest.raises(ValueError):
        await kv_manager.release(seq_id=25)

    # Release all ids
    for i, id in enumerate(seq_ids + seq_ids_2):
        await kv_manager.release(seq_id=id)
        assert kv_manager.slots_remaining == outstanding + i


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
        max_batch_size=16,
        max_seq_len=100,
        num_layers=10,
        session=session,
        device=device,
    )

    # Raise on fetch when nothing has been claimed
    with pytest.raises(ValueError):
        kv_collection = kv_manager.fetch(seq_ids=[0])

    # Claim 5 items
    seq_ids = await kv_manager.claim(5)

    # Fetch 3 of the 5 ids
    kv_collection = kv_manager.fetch(seq_ids[:3])
    assert kv_collection is not None
