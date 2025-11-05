# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
from max.dtype import DType
from max.kv_cache.paged_cache.cache_manager import PagedKVCacheManager
from max.kv_cache.paged_cache.tp_cache_manager import _TPPagedKVCacheManager
from max.nn.kv_cache import KVCacheParams, KVCacheStrategy


@pytest.mark.parametrize(
    "memory_available", [1024 * 1024 * 1024, 1024 * 1024 * 1024 * 2]
)
@pytest.mark.parametrize(
    "n_devices", [1, 2], ids=["n_devices_1", "n_devices_2"]
)
@pytest.mark.parametrize(
    "data_parallel_degree",
    [1, 2],
    ids=["data_parallel_degree_1", "data_parallel_degree_2"],
)
def test_paged_cache_manager__max_supported_sequence_length(
    memory_available: int,
    n_devices: int,
    data_parallel_degree: int,
) -> None:
    if data_parallel_degree > n_devices:
        pytest.skip(
            "Data parallel degree must be less than or equal to number of devices"
        )

    params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=8,
        head_dim=128,
        cache_strategy=KVCacheStrategy.PAGED,
        page_size=128,
        data_parallel_degree=data_parallel_degree,
        n_devices=n_devices,
    )

    num_layers = 1
    max_supported_sequence_length = (
        PagedKVCacheManager.max_supported_sequence_length(
            params, num_layers, memory_available
        )
    )

    # Normalize the values by number of devices.
    params.data_parallel_degree = 1
    bytes_per_token_per_device = (
        _TPPagedKVCacheManager.bytes_required_per_token(params, num_layers)
        // n_devices
    )
    memory_available_per_device = memory_available // n_devices

    assert (
        max_supported_sequence_length
        == memory_available_per_device // bytes_per_token_per_device
    )
    assert params.page_size is not None
    assert max_supported_sequence_length % params.page_size == 0
