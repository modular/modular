# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio

import numpy as np
from context_utils import create_text_context
from max.driver import Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheParams,
    KVCacheStrategy,
    load_kv_manager,
)


def test_kv_cache_gpu():
    asyncio.run(_test_kv_cache_gpu())


async def _test_kv_cache_gpu():
    device = Accelerator()
    kv_params = KVCacheParams(
        n_kv_heads=8,
        head_dim=128,
        dtype=DType.bfloat16,
        cache_strategy=KVCacheStrategy.CONTINUOUS,
    )
    kv_manager = load_kv_manager(
        params=kv_params,
        max_batch_size=1,
        max_seq_len=512,
        num_layers=32,
        devices=[device],
        session=InferenceSession(devices=[device]),
    )
    seq_id = kv_manager.claim(n=1)[0]
    batch = [create_text_context(seq_id, np.empty(1))]
    # suffixed [0] because we only have one device
    kv_tuple = kv_manager.fetch(batch)[0]
    assert isinstance(kv_tuple, KVCacheInputs)
    assert len(kv_tuple) == 4
