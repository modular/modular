# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
from max.dtype import DType
from max.nn.kv_cache import KVCacheParams

INF = 999999999
GIB = 1024 * 1024 * 1024


def create_params(
    dp: int = 1, tp: int = 1, page_size: int = 128
) -> KVCacheParams:
    return KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=8,
        head_dim=128,
        num_layers=1,
        page_size=page_size,
        data_parallel_degree=dp,
        n_devices=tp * dp,
    )


def test_basic() -> None:
    params = create_params()
    assert (
        params.compute_num_device_blocks(
            available_cache_memory=GIB,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == 1024
    )
    assert (
        params.estimated_memory_size(
            available_cache_memory=GIB,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == GIB
    )


def test_unaligned() -> None:
    params = create_params()
    assert (
        params.compute_num_device_blocks(
            available_cache_memory=GIB + 7,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == 1024
    )
    assert (
        params.estimated_memory_size(
            available_cache_memory=GIB + 7,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == GIB
    )


def test_big_mem() -> None:
    params = create_params()
    assert (
        params.compute_num_device_blocks(
            available_cache_memory=17 * GIB,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == 17 * 1024
    )
    assert (
        params.estimated_memory_size(
            available_cache_memory=17 * GIB,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == 17 * GIB
    )


def test_small_batch_and_seq_len() -> None:
    params = create_params()
    assert (
        params.compute_num_device_blocks(
            available_cache_memory=GIB,
            max_batch_size=4,
            max_seq_len=1000,
        )
        == 32
    )


def test_tp2() -> None:
    params = create_params(tp=2)
    assert (
        params.compute_num_device_blocks(
            available_cache_memory=GIB,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == 1024
    )
    assert (
        params.estimated_memory_size(
            available_cache_memory=GIB,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == GIB
    )


def test_limited_mem() -> None:
    params = create_params()
    with pytest.raises(
        RuntimeError,
        match="Insufficient cache memory to allocate even a single page",
    ):
        params.compute_num_device_blocks(
            available_cache_memory=1,
            max_batch_size=INF,
            max_seq_len=INF,
        )


def test_dp2() -> None:
    params = create_params(dp=2)
    assert (
        params.compute_num_device_blocks(
            available_cache_memory=GIB,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == 512
    )
    assert (
        params.estimated_memory_size(
            available_cache_memory=GIB,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == GIB
    )


def test_weird_page_size() -> None:
    params = create_params(page_size=777)
    assert (
        params.compute_num_device_blocks(
            available_cache_memory=GIB,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == 168
    )
    assert (
        params.estimated_memory_size(
            available_cache_memory=GIB,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == 1069350912
    )


def test_bytes_per_block() -> None:
    dtype = DType.float32
    n_kv_heads = 1
    head_dim = 24
    num_layers = 17
    page_size = 128
    data_parallel_degree = 1
    n_devices = 1

    params = KVCacheParams(
        dtype=dtype,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        page_size=page_size,
        data_parallel_degree=data_parallel_degree,
        n_devices=n_devices,
    )

    assert params.bytes_per_block == 417792
