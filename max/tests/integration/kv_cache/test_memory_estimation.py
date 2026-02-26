# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheParams,
    KVCacheQuantizationConfig,
    compute_num_device_blocks,
    estimated_memory_size,
)

INF = 999999999
GIB = 1024 * 1024 * 1024


def create_params(
    dp: int = 1,
    tp: int = 1,
    page_size: int = 128,
    dtype: DType = DType.float32,
    quantization_config: KVCacheQuantizationConfig = KVCacheQuantizationConfig(),
) -> KVCacheParams:
    return KVCacheParams(
        dtype=dtype,
        n_kv_heads=8,
        head_dim=128,
        num_layers=1,
        page_size=page_size,
        data_parallel_degree=dp,
        devices=[DeviceRef.GPU(i) for i in range(tp * dp)],
        kvcache_quant_config=quantization_config,
    )


def test_basic() -> None:
    params = create_params()
    assert (
        compute_num_device_blocks(
            params=params,
            available_cache_memory=GIB,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == 1024
    )
    assert (
        estimated_memory_size(
            params=params,
            available_cache_memory=GIB,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == GIB
    )


def test_unaligned() -> None:
    params = create_params()
    assert (
        compute_num_device_blocks(
            params=params,
            available_cache_memory=GIB + 7,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == 1024
    )
    assert (
        estimated_memory_size(
            params=params,
            available_cache_memory=GIB + 7,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == GIB
    )


def test_big_mem() -> None:
    params = create_params()
    assert (
        compute_num_device_blocks(
            params=params,
            available_cache_memory=17 * GIB,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == 17 * 1024
    )
    assert (
        estimated_memory_size(
            params=params,
            available_cache_memory=17 * GIB,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == 17 * GIB
    )


def test_small_batch_and_seq_len() -> None:
    params = create_params()
    assert (
        compute_num_device_blocks(
            params=params,
            available_cache_memory=GIB,
            max_batch_size=4,
            max_seq_len=1000,
        )
        == 32
    )


def test_tp2() -> None:
    params = create_params(tp=2)
    assert (
        compute_num_device_blocks(
            params=params,
            available_cache_memory=GIB,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == 1024
    )
    assert (
        estimated_memory_size(
            params=params,
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
        compute_num_device_blocks(
            params=params,
            available_cache_memory=1,
            max_batch_size=INF,
            max_seq_len=INF,
        )


def test_dp2() -> None:
    params = create_params(dp=2)
    assert (
        compute_num_device_blocks(
            params=params,
            available_cache_memory=GIB,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == 512
    )
    assert (
        estimated_memory_size(
            params=params,
            available_cache_memory=GIB,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == GIB
    )


def test_weird_page_size() -> None:
    params = create_params(page_size=777)
    assert (
        compute_num_device_blocks(
            params=params,
            available_cache_memory=GIB,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == 168
    )
    assert (
        estimated_memory_size(
            params=params,
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
        devices=[DeviceRef.GPU()],
    )

    assert params.bytes_per_block == 417792


def test_quantized_kv_cache() -> None:
    fp8_params = create_params(dp=2, dtype=DType.float8_e4m3fn)
    fp32_params = create_params(dp=2, dtype=DType.float32)

    fp8_estimated_memory = estimated_memory_size(
        params=fp8_params,
        available_cache_memory=GIB,
        max_batch_size=INF,
        max_seq_len=INF,
    )
    fp32_estimated_memory = estimated_memory_size(
        params=fp32_params,
        available_cache_memory=GIB,
        max_batch_size=INF,
        max_seq_len=INF,
    )
    assert fp32_estimated_memory == GIB
    # The estimated memory is the same in both cases since more blocks can be allocated when the KVCache is compressed.
    assert fp8_estimated_memory < fp32_estimated_memory

    # The number of bytes / block should be ~4x smaller for FP8 KVCache (compared to FP32).
    assert fp8_params.bytes_per_block < fp32_params.bytes_per_block

    fp8_num_device_blocks = compute_num_device_blocks(
        params=fp8_params,
        available_cache_memory=GIB,
        max_batch_size=INF,
        max_seq_len=INF,
    )
    fp32_num_device_blocks = compute_num_device_blocks(
        params=fp32_params,
        available_cache_memory=GIB,
        max_batch_size=INF,
        max_seq_len=INF,
    )
    # Nearly ~4x more blocks can be allocated when using FP8 KVCache (compared to FP32).
    assert fp8_num_device_blocks > fp32_num_device_blocks
