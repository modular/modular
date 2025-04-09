# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from max.driver import CPU
from max.dtype import DType
from max.nn.kv_cache import KVCacheParams, KVCacheStrategy, PagedKVCacheManager


def test_paged_kv_cache_memory_estimates_truncated():
    kv_params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=8,
        head_dim=128,
        cache_strategy=KVCacheStrategy.PAGED,
        page_size=128,
    )
    available_cache_memory = 10 * 1024 * 1024 * 1024
    max_seq_len = 1024
    max_batch_size = 1

    estimated_cache_memory = PagedKVCacheManager.estimated_memory_size(
        kv_params,
        max_batch_size,
        max_seq_len,
        1,
        available_cache_memory,
        [CPU(0)],
    )

    expected_value = 1024 * 8 * 128 * 2 * 4
    assert estimated_cache_memory == expected_value


def test_paged_kv_cache_memory_estimates_not_truncated():
    kv_params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=8,
        head_dim=128,
        cache_strategy=KVCacheStrategy.PAGED,
        page_size=128,
    )
    available_cache_memory = (
        8000000  # notably less than the calculated size of the cache.
    )
    max_seq_len = 1024
    max_batch_size = 1

    estimated_cache_memory = PagedKVCacheManager.estimated_memory_size(
        kv_params,
        max_batch_size,
        max_seq_len,
        1,
        available_cache_memory,
        [CPU(0)],
    )

    assert estimated_cache_memory == available_cache_memory
