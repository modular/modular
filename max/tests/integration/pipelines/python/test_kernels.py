# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph
from nn.kernels import kv_cache_length
from nn.kv_cache import (
    ContiguousKVCacheCollectionType,
    ContiguousKVCacheManager,
)
from nn.kv_caching import KVCacheParams


def test_kv_cache_length():
    asyncio.run(_test_kv_cache_length())


async def _test_kv_cache_length() -> None:
    """Tests that KV cache collections return the expected cache length."""
    kv_params = KVCacheParams(
        dtype=DType.bfloat16, n_kv_heads=8, head_dim=128, device=CPU()
    )
    session = InferenceSession()
    kv_manager = ContiguousKVCacheManager(
        params=kv_params,
        max_batch_size=1,
        max_seq_len=512,
        num_layers=32,
        session=session,
        device=CPU(),
    )

    # Reserve a slot in the KV cache manager.
    seq_id = await kv_manager.claim(batch_size=1)
    seq_id = seq_id[0]

    # Set the cache lengths first by "stepping".
    expected_cache_len = 42
    kv_manager.step(valid_lengths={seq_id: expected_cache_len})

    # Construct a KV cache collection with the given cache length.
    kv_collection = kv_manager.fetch(seq_ids=[seq_id])
    assert kv_collection is not None

    graph = Graph(
        "cache_length",
        lambda kv_coll: kv_cache_length(kv_params, kv_coll),
        input_types=[ContiguousKVCacheCollectionType()],
    )

    outputs = session.load(graph).execute(kv_collection)
    assert outputs[0].item() == expected_cache_len
