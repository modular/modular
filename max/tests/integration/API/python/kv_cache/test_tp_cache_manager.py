# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import pytest
from max.dtype import DType
from max.kv_cache.paged_cache.tp_cache_manager import _TPPagedKVCacheManager
from max.nn.kv_cache import KVCacheParams, KVCacheStrategy


def test_tp_cache_manager__bytes_required_per_token() -> None:
    params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=8,
        head_dim=128,
        cache_strategy=KVCacheStrategy.PAGED,
        page_size=128,
    )

    num_layers = 1
    bytes_required_per_token = _TPPagedKVCacheManager.bytes_required_per_token(
        params, num_layers
    )
    manual_bytes_required_per_token = 2 * 8 * 128 * num_layers * 4
    assert bytes_required_per_token == manual_bytes_required_per_token


@pytest.mark.parametrize(
    "memory_available", [1024 * 1024 * 1024, 1024 * 1024 * 1024 * 2]
)
def test_tp_cache_manager__max_supported_sequence_length(
    memory_available: int,
) -> None:
    params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=8,
        head_dim=128,
        cache_strategy=KVCacheStrategy.PAGED,
        page_size=128,
    )

    num_layers = 1
    max_supported_sequence_length = (
        _TPPagedKVCacheManager.max_supported_sequence_length(
            params, num_layers, memory_available
        )
    )
    bytes_required_per_token = _TPPagedKVCacheManager.bytes_required_per_token(
        params, num_layers
    )

    assert (
        max_supported_sequence_length
        <= memory_available // bytes_required_per_token
    )
    assert params.page_size is not None
    assert max_supported_sequence_length % params.page_size == 0
