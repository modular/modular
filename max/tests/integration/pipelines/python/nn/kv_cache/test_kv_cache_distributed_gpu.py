# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio

import numpy as np
from max.driver import Accelerator, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.pipelines.kv_cache import (
    KVCacheInputs,
    KVCacheManager,
    KVCacheParams,
    KVCacheStrategy,
    load_kv_manager,
)
from test_common.context_utils import create_text_context


def test_kv_cache_gpu():
    asyncio.run(_test_kv_cache_gpu())


async def _test_kv_cache_gpu():
    num_devices = accelerator_count()

    if num_devices > 1:
        list_of_devices = [Accelerator(id=i) for i in range(num_devices)]
        inference_session = InferenceSession(devices=list_of_devices)
        kv_params = KVCacheParams(
            n_kv_heads=8,
            head_dim=128,
            dtype=DType.bfloat16,
            cache_strategy=KVCacheStrategy.CONTINUOUS,
            n_devices=num_devices,
        )
        kv_manager: KVCacheManager = load_kv_manager(
            params=kv_params,
            max_batch_size=1,
            max_seq_len=512,
            num_layers=32,
            devices=list_of_devices,
            session=inference_session,
        )
        seq_id = kv_manager.claim(n=1)[0]

        batch = [create_text_context(seq_id, np.empty(1))]
        list_of_kv_tuples = kv_manager.fetch(batch)
        for i in range(num_devices):
            kv_tuple = list_of_kv_tuples[i]
            assert isinstance(kv_tuple, KVCacheInputs)
            assert len(kv_tuple) == 4
