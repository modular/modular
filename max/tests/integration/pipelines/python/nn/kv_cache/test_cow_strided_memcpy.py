# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import pytest
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.pipelines.kv_cache import (
    KVCacheParams,
    KVCacheStrategy,
    PagedKVCacheManager,
)


def create_paged_manager(
    num_blocks: int,
    page_size: int = 1,
    num_kv_heads: int = 1,
    head_dim: int = 1,
    num_layers: int = 1,
) -> PagedKVCacheManager:
    kv_params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=num_kv_heads,
        head_dim=head_dim,
        cache_strategy=KVCacheStrategy.PAGED,
        enable_prefix_caching=True,
        page_size=page_size,
    )

    session = InferenceSession()

    cache_memory = (
        2
        * num_layers
        * num_kv_heads
        * head_dim
        * page_size
        * num_blocks
        * kv_params.dtype.size_in_bytes
    )
    kv_manager = PagedKVCacheManager(
        params=kv_params,
        max_batch_size=1,
        max_seq_len=1,
        num_layers=num_layers,
        devices=[CPU()],
        session=session,
        cache_memory=cache_memory,
        page_size=page_size,
    )

    assert len(kv_manager.available_blocks) == num_blocks
    return kv_manager


@pytest.mark.asyncio
async def test_cow_strided_memcpy() -> None:
    """Tests that KV cache collections return the expected cache length."""

    paged_manager = create_paged_manager(
        num_blocks=10, page_size=128, num_kv_heads=3, head_dim=4, num_layers=5
    )
    assert paged_manager.prefix_cache is not None
    assert paged_manager.prefix_cache.cow_strided_memcpy_graph is not None

    cow_graph = paged_manager.prefix_cache.cow_strided_memcpy_graph
    blocks = paged_manager.blocks[0]

    for (
        num_layers_idx,
        kv_dim_idx,
        block_idx,
        token_idx,
        n_kv_heads_per_device_idx,
        head_dim_idx,
    ) in blocks._iterate_indices():
        blocks[
            num_layers_idx,
            kv_dim_idx,
            block_idx,
            token_idx,
            n_kv_heads_per_device_idx,
            head_dim_idx,
        ] = 10 * block_idx + token_idx

    # copy the contents of the first 64 of 128 tokens in the 3rd block to the 9th block
    block_dst_idx = 9
    block_src_idx = 3
    num_tokens = 64
    cow_graph.execute(block_dst_idx, block_src_idx, num_tokens, blocks)

    blocks_cpu_after = blocks.to_numpy()
    for (
        num_layers_idx,
        kv_dim_idx,
        block_idx,
        token_idx,
        n_kv_heads_per_device_idx,
        head_dim_idx,
    ) in blocks._iterate_indices():
        x = blocks_cpu_after[
            num_layers_idx,
            kv_dim_idx,
            block_idx,
            token_idx,
            n_kv_heads_per_device_idx,
            head_dim_idx,
        ]
        if block_idx == block_dst_idx and token_idx < num_tokens:
            # these tokens in the 9th block should match the tokens in the 3rd block
            assert x == 10 * block_src_idx + token_idx
        else:
            # all other tokens should be unchanged
            assert x == 10 * block_idx + token_idx
