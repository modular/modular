# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType
from max.pipelines.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    KVCacheParams,
    KVCacheStrategy,
    load_kv_manager,
)


@pytest.mark.asyncio
async def test_kv_collection_constructor_continuous() -> None:
    """Tests that KV cache collections return the expected cache length."""
    kv_params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=8,
        head_dim=128,
        cache_strategy=KVCacheStrategy.CONTINUOUS,
    )
    session = InferenceSession()
    kv_manager = load_kv_manager(
        params=kv_params,
        max_cache_batch_size=1,
        max_seq_len=512,
        num_layers=32,
        devices=[CPU()],
        session=session,
    )

    # Reserve a slot in the KV cache manager.
    seq_id = kv_manager.claim(n=1)[0]

    # Set the cache lengths first by "stepping".
    expected_cache_len = 42
    kv_manager.step(valid_lengths={seq_id: expected_cache_len})

    # Construct a KV cache collection with the given cache length.
    kv_tuple = kv_manager.fetch(seq_ids=[seq_id])
    assert len(kv_tuple) == 4

    graph = Graph(
        "create_collection",
        FetchContinuousBatchingKVCacheCollection(kv_params),
        input_types=kv_manager.input_symbols(),
    )

    outputs = session.load(graph).execute(*kv_tuple)
    kv_collection = outputs[0]
    assert kv_collection is not None
