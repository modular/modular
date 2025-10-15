# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
from max.dtype import DType
from max.nn.kv_cache.cache_params import KVCacheParams


def test_single_device_compatible() -> None:
    """Test single device configuration (no DP or TP)."""
    params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=128,
        n_devices=1,
        data_parallel_degree=1,
        page_size=16,
    )
    assert params.n_kv_heads_per_device == 8


def test_tensor_parallel_compatible_divisible_heads() -> None:
    """Test TP mode with n_kv_heads divisible by n_devices."""
    params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=128,
        n_devices=2,
        data_parallel_degree=1,
        page_size=16,
    )
    assert params.n_kv_heads_per_device == 4


def test_tensor_parallel_compatible_multiple_devices() -> None:
    """Test TP mode with 4 devices and 16 heads."""
    params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=16,
        head_dim=128,
        n_devices=4,
        data_parallel_degree=1,
        page_size=16,
    )
    assert params.n_kv_heads_per_device == 4


def test_tensor_parallel_compatible_large_heads() -> None:
    """Test TP mode with many heads evenly distributed."""
    params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=32,
        head_dim=128,
        n_devices=8,
        data_parallel_degree=1,
        page_size=16,
    )
    assert params.n_kv_heads_per_device == 4


def test_data_parallel_compatible_equal_devices() -> None:
    """Test DP mode with data_parallel_degree equal to n_devices."""
    params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=128,
        n_devices=4,
        data_parallel_degree=4,
        page_size=16,
    )
    # In DP mode, heads are not sharded
    assert params.n_kv_heads_per_device == 8


def test_data_parallel_compatible_multiple_devices() -> None:
    """Test DP mode with multiple devices."""
    params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=12,
        head_dim=64,
        n_devices=2,
        data_parallel_degree=2,
        page_size=16,
    )
    # In DP mode, all heads are on each device
    assert params.n_kv_heads_per_device == 12


# ==================== Incompatible Cases ====================


def test_data_parallel_exceeds_devices_fails() -> None:
    """Test that DP degree > n_devices raises ValueError."""
    with pytest.raises(
        ValueError,
        match=r"Data parallelism degree \(4\) cannot be greater than the number of devices \(2\)",
    ):
        KVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=8,
            head_dim=128,
            n_devices=2,
            data_parallel_degree=4,
            page_size=16,
        )


def test_data_parallel_exceeds_devices_large_degree_fails() -> None:
    """Test that DP degree >> n_devices raises ValueError."""
    with pytest.raises(
        ValueError,
        match=r"Data parallelism degree \(8\) cannot be greater than the number of devices \(1\)",
    ):
        KVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=16,
            head_dim=128,
            n_devices=1,
            data_parallel_degree=8,
            page_size=16,
        )


def test_mixed_dp_tp_not_supported_fails() -> None:
    """Test that DP + TP combination is not yet supported."""
    with pytest.raises(
        ValueError,
        match=r"We do not yet support DP \+ TP at the same time.*data_parallel_degree=2.*n_devices=4",
    ):
        KVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=8,
            head_dim=128,
            n_devices=4,
            data_parallel_degree=2,
            page_size=16,
        )


def test_mixed_dp_tp_another_combination_fails() -> None:
    """Test another DP + TP combination that should fail."""
    with pytest.raises(
        ValueError,
        match=r"We do not yet support DP \+ TP at the same time.*data_parallel_degree=3.*n_devices=6",
    ):
        KVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=12,
            head_dim=64,
            n_devices=6,
            data_parallel_degree=3,
            page_size=16,
        )


def test_tensor_parallel_non_divisible_heads_fails() -> None:
    """Test that TP mode with non-divisible heads raises ValueError."""
    with pytest.raises(
        ValueError,
        match=r"Number of KV heads \(8\) must be divisible by the number of devices \(3\)",
    ):
        KVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=8,
            head_dim=128,
            n_devices=3,
            data_parallel_degree=1,
            page_size=16,
        )


def test_tensor_parallel_non_divisible_heads_small_fails() -> None:
    """Test TP mode where n_kv_heads < n_devices."""
    with pytest.raises(
        ValueError,
        match=r"Number of KV heads \(2\) must be divisible by the number of devices \(4\)",
    ):
        KVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=2,
            head_dim=128,
            n_devices=4,
            data_parallel_degree=1,
            page_size=16,
        )


def test_tensor_parallel_odd_division_fails() -> None:
    """Test TP mode with an odd number that doesn't divide evenly."""
    with pytest.raises(
        ValueError,
        match=r"Number of KV heads \(7\) must be divisible by the number of devices \(2\)",
    ):
        KVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=7,
            head_dim=128,
            n_devices=2,
            data_parallel_degree=1,
            page_size=16,
        )
