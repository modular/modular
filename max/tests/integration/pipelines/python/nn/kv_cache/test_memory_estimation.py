# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
from max.driver import CPU
from max.dtype import DType
from max.nn.kv_cache import KVCacheParams, KVCacheStrategy, PagedKVCacheManager
from max.support.math import ceildiv


@pytest.mark.parametrize(
    "tp,dp", [(1, 1), (1, 2), (2, 1), (2, 2), (4, 2), (2, 4), (1, 8), (8, 1)]
)
def test_memory_estimation(tp: int, dp: int) -> None:
    page_size = 128
    n_kv_heads = 8
    head_dim = 128
    num_layers = 10
    dtype = DType.float32
    devices = [CPU(0) for _ in range(tp * dp)]
    bytes_per_page_per_replica = (
        n_kv_heads * head_dim * page_size * num_layers * dtype.size_in_bytes
    )
    available_cache_memory = 1024 * 1024 * 1024
    max_seq_len = 1024
    max_batch_size = 32
    available_cache_memory_per_replica = available_cache_memory // dp
    max_pages_per_replica = (
        available_cache_memory_per_replica // bytes_per_page_per_replica
    )
    blocks_per_max_seq_len = ceildiv(max_seq_len, page_size)
    max_pages_per_replica = min(
        max_pages_per_replica,
        blocks_per_max_seq_len * max_batch_size,
    )
    params = KVCacheParams(
        dtype=dtype,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        cache_strategy=KVCacheStrategy.PAGED,
        page_size=page_size,
    )
    memory_size = PagedKVCacheManager.estimated_memory_size(
        params=params,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        available_cache_memory=available_cache_memory,
        devices=devices,
    )
    # This test may compute a slightly different memory size than estimated_memory_size().
    # The calculation done here aligns the memory_size downward to be a multiple
    # of bytes per page.
    assert (
        max_pages_per_replica * bytes_per_page_per_replica * dp
        <= memory_size
        <= available_cache_memory
    )
