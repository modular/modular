# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import numpy as np
import pytest
from max.driver import Accelerator, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheManager,
    KVCacheParams,
    KVCacheStrategy,
    PagedKVCacheManager,
    load_kv_manager,
)
from max.pipelines.core import InputContext
from test_common.context_utils import create_text_context


@pytest.mark.asyncio
async def test_kv_cache_multi_gpu():
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


def create_paged_manager(
    num_blocks: int,
    max_batch_size: int,
    max_seq_len: int,
    page_size: int,
    enable_prefix_caching: bool = False,
    enable_kvcache_swapping_to_host: bool = False,
) -> PagedKVCacheManager:
    NUM_KV_HEADS = 1
    HEAD_DIM = 1
    NUM_LAYERS = 1

    dtype = DType.float32

    devices = [Accelerator(id=i) for i in range(accelerator_count())]

    cache_memory = (
        2
        * NUM_LAYERS
        * NUM_KV_HEADS
        * HEAD_DIM
        * page_size
        * num_blocks
        * dtype.size_in_bytes
        * len(devices)
    )

    # CPU swap space is 100x the device cache memory
    GiB = 1024 * 1024 * 1024
    host_kvcache_swap_space_gb = 100 * cache_memory / GiB

    kv_params = KVCacheParams(
        dtype=dtype,
        n_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        cache_strategy=KVCacheStrategy.PAGED,
        page_size=page_size,
        enable_prefix_caching=enable_prefix_caching,
        enable_kvcache_swapping_to_host=enable_kvcache_swapping_to_host,
        host_kvcache_swap_space_gb=host_kvcache_swap_space_gb,
    )

    session = InferenceSession(devices=devices)

    kv_manager = PagedKVCacheManager(
        params=kv_params,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        num_layers=NUM_LAYERS,
        devices=devices,
        session=session,
        cache_memory=cache_memory,
        page_size=page_size,
        enable_runtime_checks=True,
    )

    assert kv_manager.total_num_pages == num_blocks
    return kv_manager


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_swapping_to_host", [True, False])
async def test_swapping_to_host_multi_gpu(enable_swapping_to_host: bool):
    # set seed for reproducibility
    np.random.seed(42)

    # Enough blocks to hold 500 tokens
    kv_manager = create_paged_manager(
        num_blocks=100,
        max_batch_size=100,
        max_seq_len=512,
        page_size=5,
        enable_prefix_caching=True,
        enable_kvcache_swapping_to_host=enable_swapping_to_host,
    )

    if enable_swapping_to_host:
        # Host tensor should be pinned
        assert kv_manager.host_tensor is not None
        assert kv_manager.host_tensor.pinned
        # Evictions should be scheduled on auxiliary stream
        assert kv_manager.block_manager.block_copy_engine is not None
        assert kv_manager.block_manager.block_copy_engine.supports_multistream()

    def gen_prompt(length: int) -> np.ndarray:
        # returns a binary sequence of length `length`
        return np.random.randint(0, 2, size=length)

    # There are 20 requests.
    # Each request is 100 tokens so there are 2000 tokens.
    # The last 10 requests are duplicates of the first 10.
    # Since the last 10 reqs are duplicates, we need approximately 1000 tokens worth of blocks.
    # This exceeds the 500 token limit so we will need to swap to host.
    prompt_len = 100
    reqs: list[InputContext] = []
    for i in range(10):
        reqs.append(create_text_context(len(reqs), gen_prompt(prompt_len)))
    for i in range(10):
        reqs.append(create_text_context(len(reqs), reqs[i].tokens))

    # Each batch has 4 requests
    batch_size = 4
    batches: list[list[InputContext]] = [
        reqs[i : i + batch_size] for i in range(0, len(reqs), batch_size)
    ]

    cache_hit_rates = []
    for batch_idx, batch in enumerate(batches):
        seq_ids = [ctx.cache_seq_id for ctx in batch]
        kv_manager.external_claim(seq_ids)

        # Run 1 CE batch and 4 TG batches
        for iter in range(5):
            prompt_tokens = sum(ctx.active_length for ctx in batch)

            _ = kv_manager.fetch(batch)

            new_prompt_tokens = sum(ctx.active_length for ctx in batch)

            # Check cache hit rate for the first iteration (CE)
            if iter == 0:
                cached_tokens = prompt_tokens - new_prompt_tokens
                pct = cached_tokens / prompt_tokens
                cache_hit_rates.append(pct)
                print(
                    f"[Batch {batch_idx}] Hit rate: {cached_tokens} / {prompt_tokens} = {pct:.2%}"
                )

            for ctx in batch:
                ctx.update(999)

            kv_manager.step(batch)

        for seq_id in seq_ids:
            kv_manager.release(seq_id)

    if enable_swapping_to_host:
        # cache hit rates are high!
        expected_cache_hit_rates = np.array([0.0, 0.025, 0.49, 0.95, 0.95])
        expected_blocks_copied = np.array([3, 199, 190])  # d2d, d2h, h2d
    else:
        # cache hit rate are very low :(
        expected_cache_hit_rates = np.array([0.0, 0.02, 0.03, 0.02, 0.03])
        expected_blocks_copied = np.array([11, 0, 0])  # d2d, d2h, h2d

    blocks_copied = kv_manager.num_blocks_copied
    print(
        f"Blocks copied: D2D: {blocks_copied.d2d}, D2H: {blocks_copied.d2h}, H2D: {blocks_copied.h2d}"
    )
    blocks_copied_arr = np.array(
        [blocks_copied.d2d, blocks_copied.d2h, blocks_copied.h2d]
    )
    assert np.allclose(blocks_copied_arr, expected_blocks_copied, atol=5)
    assert np.allclose(cache_hit_rates, expected_cache_hit_rates, atol=0.1)
