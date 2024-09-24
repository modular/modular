# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession

from nn.kv_cache import ContiguousKVCacheManager
from nn.kv_cache_params import KVCacheParams


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

    # Claim one and evaluate the list returned.
    seq_ids = kv_manager.claim(batch_size=1)
    assert len(seq_ids) == 1

    # Claim four and evaluate the list returned.
    seq_ids_2 = kv_manager.claim(batch_size=4)
    assert len(seq_ids_2) == 4

    # Assert that we are not claiming the same seq_ids twice.
    assert len(set(seq_ids).intersection(set(seq_ids_2))) == 0

    kv_collection = kv_manager.fetch(seq_ids + seq_ids_2)
    assert kv_collection is not None
