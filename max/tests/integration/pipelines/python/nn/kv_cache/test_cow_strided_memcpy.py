# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import numpy as np
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

    assert kv_manager.total_num_pages == num_blocks
    return kv_manager


def cow_strided_memcpy_reference(
    blocks: np.ndarray,
    block_dst_idx_tensor: np.ndarray,
    block_src_idx_tensor: np.ndarray,
    num_tokens_tensor: np.ndarray,
    _max_num_tokens_scalar: np.uint32,
) -> None:
    batch_dim = num_tokens_tensor.shape[0]
    for batch_idx in range(batch_dim):
        block_dst_idx = block_dst_idx_tensor[batch_idx]
        block_src_idx = block_src_idx_tensor[batch_idx]
        num_tokens = num_tokens_tensor[batch_idx]
        blocks[block_dst_idx, :, :, :num_tokens, :, :] = blocks[
            block_src_idx, :, :, :num_tokens, :, :
        ]


@pytest.mark.asyncio
async def test_cow_strided_memcpy() -> None:
    """Tests that KV cache collections return the expected cache length."""

    paged_manager = create_paged_manager(
        num_blocks=10, page_size=128, num_kv_heads=3, head_dim=4, num_layers=5
    )
    cow_graph = paged_manager.cow_executor.cow_strided_memcpy_model
    assert cow_graph is not None
    blocks = paged_manager.tensors[0]

    # initialize the blocks with some values
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

    # copy the contents of the first 64 tokens in the 3rd block to the 9th block
    # copy the contents of the first 23 tokens in the 3rd block to the 4th block
    # copy the contents of the first 41 tokens in the 2rd block to the 6th block
    # ...
    block_dst_idx_tensor = np.array([9, 4, 6, 7, 0], dtype=np.uint32)
    block_src_idx_tensor = np.array([3, 3, 2, 2, 8], dtype=np.uint32)
    num_tokens_tensor = np.array([64, 23, 41, 72, 128], dtype=np.uint32)
    max_num_tokens_scalar = np.max(num_tokens_tensor)

    # compute the expected result
    blocks_expected: np.ndarray = blocks.to_numpy().copy()
    cow_strided_memcpy_reference(
        blocks_expected,
        block_dst_idx_tensor,
        block_src_idx_tensor,
        num_tokens_tensor,
        max_num_tokens_scalar,
    )

    # compute the actual result
    cow_graph.execute(
        block_dst_idx_tensor,
        block_src_idx_tensor,
        num_tokens_tensor,
        max_num_tokens_scalar,
        blocks,
    )
    blocks_actual = blocks.to_numpy()

    # check that the actual result matches the expected result
    assert np.allclose(blocks_actual, blocks_expected)
