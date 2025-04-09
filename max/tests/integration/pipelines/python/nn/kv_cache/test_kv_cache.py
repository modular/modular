# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import numpy as np
import pytest
from context_utils import create_text_context
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph
from max.nn.kv_cache import (
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
        page_size=128,
    )

    kv_manager_kwargs = {}
    if cache_strategy == KVCacheStrategy.PAGED:
        kv_manager_kwargs["page_size"] = 128

    session = InferenceSession()

    # let's set an arbitrary 500 Mb allocation
    available_cache_memory = 500 * 2**20
    kv_manager = load_kv_manager(
        params=kv_params,
        max_batch_size=1,
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
    batch = [create_text_context(seq_id, np.empty(expected_cache_len))]

    kv_manager.external_claim(seq_ids=[seq_id])
    kv_tuple_list = kv_manager.fetch(batch)

    # Set the cache lengths first by "stepping".
    batch[0].update(42)
    kv_manager.step(batch)

    # Construct a KV cache collection with the given cache length.
    kv_tuple_list = kv_manager.fetch(batch)
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
