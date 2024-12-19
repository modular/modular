# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph
from max.pipelines.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    KVCacheParams,
    KVCacheStrategy,
    load_kv_manager,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cache_strategy, fetch_cls",
    [
        (KVCacheStrategy.CONTINUOUS, FetchContinuousBatchingKVCacheCollection),
        (KVCacheStrategy.PAGED, FetchPagedKVCacheCollection),
    ],
)
async def test_kv_collection_constructor(cache_strategy, fetch_cls) -> None:
    """Tests that KV cache collections return the expected cache length."""
    kv_params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=8,
        head_dim=128,
        cache_strategy=cache_strategy,
    )

    kv_manager_kwargs = {}
    if cache_strategy == KVCacheStrategy.PAGED:
        kv_manager_kwargs["page_size"] = 128

    session = InferenceSession()

    # let's set an arbitrary 100 Mb allocation
    available_cache_memory = 100 * 2**20
    kv_manager = load_kv_manager(
        params=kv_params,
        max_cache_batch_size=1,
        max_seq_len=512,
        num_layers=32,
        devices=[CPU()],
        session=session,
        available_cache_memory=available_cache_memory,
        **kv_manager_kwargs,
    )

    # Reserve a slot in the KV cache manager.
    seq_id = 0
    expected_cache_len = 42
    seq_ids_and_lengths = {seq_id: expected_cache_len}

    kv_manager.external_claim(seq_ids=[seq_id])
    kv_tuple_list = kv_manager.fetch(seq_ids_and_lengths=seq_ids_and_lengths)

    # Set the cache lengths first by "stepping".
    kv_manager.step(seq_ids_and_lengths=seq_ids_and_lengths)

    # Construct a KV cache collection with the given cache length.
    cache_lengths = {seq_id: 1}
    kv_tuple_list = kv_manager.fetch(cache_lengths)
    assert len(kv_tuple_list) == 1
    assert len(kv_tuple_list[0]) == 4
    kv_tuple = kv_tuple_list[0]

    graph = Graph(
        "create_collection",
        fetch_cls(kv_params),
        input_types=kv_manager.input_symbols()[0],
    )

    outputs = session.load(graph).execute(*kv_tuple)
    kv_collection = outputs[0]
    assert kv_collection is not None
