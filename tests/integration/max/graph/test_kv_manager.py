# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from nn.kv_cache import KVCacheParams, ContiguousKVCacheManager


async def _claim_fetch_and_reset(kv_manager):
    # Claim one and evaluate the list returned
    seq_ids = await kv_manager.claim(batch_size=1)
    assert len(seq_ids) == 1

    # Claim four and evaluate the list returned.
    seq_ids_2 = await kv_manager.claim(batch_size=4)
    assert len(seq_ids_2) == 4

    # Assert that we are not claiming the same seq_ids twice.
    assert len(set(seq_ids).intersection(set(seq_ids_2))) == 0

    kv_collection = kv_manager.fetch(seq_ids + seq_ids_2)
    assert kv_collection is not None

    # Check that resetting the cache succeeds.
    await kv_manager.reset_cache()

    # Check that the update function operates as intended.
    seq_ids = await kv_manager.claim(batch_size=3)
    _ = kv_manager.fetch(seq_ids)

    for i in range(3):
        valid_lengths = {}
        for seq_id in seq_ids:
            valid_lengths[seq_id] = 1

        kv_manager.step(valid_lengths)

        for seq_id in seq_ids:
            assert kv_manager.cache_lengths[seq_id] == 1 + i

    # Reset the cache, claim 4 ids, and pass 3 to the fetch.
    # This tests that the cache_lengths is appropriately pulled.
    await kv_manager.reset_cache()

    seq_ids = await kv_manager.claim(batch_size=4)
    kv_collection = kv_manager.fetch(seq_ids[1:])

    await kv_manager.reset_cache()


async def _claim_and_release(kv_manager: ContiguousKVCacheManager):
    # Claim 5 ids
    seq_ids = await kv_manager.claim(batch_size=5)

    outstanding = 11

    # Introspect to ensure that we've only reserved 5
    # Max Batch size is 16, therefore the remaining is 11
    assert kv_manager.semaphore._value == outstanding

    for seq_id in seq_ids:
        await kv_manager.release(seq_id)
        outstanding += 1
        assert kv_manager.semaphore._value == outstanding

    await kv_manager.reset_cache()


def test_kv_manager(session: InferenceSession) -> None:
    # Initialize llama like params.
    params = KVCacheParams(
        dtype=DType.float32, n_kv_heads=8, head_dim=128, device=CPU()
    )

    kv_manager = ContiguousKVCacheManager(
        params=params,
        num_layers=10,
        max_batch_size=16,
        max_seq_len=100,
        session=session,
        device=CPU(),
    )

    asyncio.run(_claim_fetch_and_reset(kv_manager))

    asyncio.run(_claim_and_release(kv_manager))
