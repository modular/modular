# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
from max.driver import CUDA
from max.dtype import DType
from max.engine import InferenceSession
from nn.kv_cache import KVCacheParams, KVCacheStrategy, load_kv_manager


def test_kv_cache_gpu():
    asyncio.run(_test_kv_cache_gpu())


async def _test_kv_cache_gpu():
    device = CUDA()
    kv_params = KVCacheParams(
        n_kv_heads=8,
        head_dim=128,
        dtype=DType.bfloat16,
        device=device,
        cache_strategy=KVCacheStrategy.CONTINUOUS,
    )
    session = InferenceSession(device=device)
    kv_manager = load_kv_manager(
        params=kv_params,
        max_cache_batch_size=1,
        max_seq_len=512,
        num_layers=32,
        session=session,
        device=device,
    )
    seq_id = await kv_manager.claim(n=1)
    seq_id = seq_id[0]
    kv_tuple = kv_manager.fetch([seq_id])
    assert isinstance(kv_tuple, tuple)
    assert len(kv_tuple) == 4
