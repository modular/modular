# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from max.driver import CUDA
from max.dtype import DType
from max.engine import InferenceSession
from nn.kv_cache import ContiguousKVCacheManager
from nn.kv_cache_params import KVCacheParams


def test_kv_cache_gpu():
    device = CUDA()
    kv_params = KVCacheParams(
        n_kv_heads=8,
        head_dim=128,
        dtype=DType.bfloat16,
        device=device,
    )
    session = InferenceSession(device=device)
    kv_manager = ContiguousKVCacheManager(
        params=kv_params,
        max_batch_size=1,
        max_seq_len=512,
        num_layers=32,
        session=session,
        device=device,
    )
    seq_id = kv_manager.claim(batch_size=1)[0]
    kv_collection = kv_manager.fetch([seq_id])
    assert kv_collection is not None
