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

from __future__ import annotations

import pytest
from max._kv_cache_ops import MLASparseKVForm
from max.driver import accelerator_api
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheQuantizationConfig,
    MLAKVCacheParams,
    MultiKVCacheParams,
)
from max.nn.kv_cache.cache_params import _SPARSE_MLA_KEY_SATURATION
from max.nn.kv_cache.utils import MLAAttnKey


@pytest.fixture(scope="module")
def gpu_device_ref() -> DeviceRef:
    return DeviceRef.GPU()


# (batch_size, num_heads, sparse_indices_stride, AMD partitions for an FP8
# cache, AMD partitions for a BF16 cache). The counts hold for 256 and 304
# compute units alike. FP8 reads the rows in place, one wave of CTAs of at
# least 32 rows; BF16 gathers them, two waves of 128-row tiles, so a
# 2051-row selection (k-pool tail, 17 tiles) takes 32 partitions below batch
# 32. 65536 rows at batch 1 clamp to the reducer's 256.
SPARSE_MLA_CASES = [
    (1, 8, 2048, 64, 16),
    (16, 8, 2048, 16, 16),
    (32, 8, 2048, 8, 16),
    (64, 8, 2048, 4, 8),
    (128, 8, 2048, 2, 4),
    (16, 16, 2048, 16, 16),
    (16, 8, 2051, 16, 32),
    (32, 8, 2051, 8, 16),
    (1, 8, 65536, 256, 256),
]
SPARSE_MLA_CACHE_LENGTHS = [128, 2048, 8192, 100_000]


@pytest.mark.parametrize(
    (
        "batch_size",
        "num_heads",
        "sparse_indices_stride",
        "fp8_partitions",
        "bf16_partitions",
    ),
    SPARSE_MLA_CASES,
)
@pytest.mark.parametrize("dtype", [DType.float8_e4m3fn, DType.bfloat16])
def test_sparse_mla_dispatch_key_follows_selection(
    gpu_device_ref: DeviceRef,
    batch_size: int,
    num_heads: int,
    sparse_indices_stride: int,
    fp8_partitions: int,
    bf16_partitions: int,
    dtype: DType,
) -> None:
    amd_partitions = (
        fp8_partitions if dtype == DType.float8_e4m3fn else bf16_partitions
    )

    def params(stride: int | None) -> MLAKVCacheParams:
        return MLAKVCacheParams(
            dtype=dtype,
            head_dim=576,
            num_layers=1,
            devices=[gpu_device_ref],
            num_q_heads=num_heads,
            sparse_indices_stride=stride,
        )

    sparse = params(sparse_indices_stride)
    multi = MultiKVCacheParams.from_params({"mla": sparse, "indexer": sparse})
    if accelerator_api() == "hip":
        for cache_length in SPARSE_MLA_CACHE_LENGTHS:
            key = sparse.resolve_attn_key(batch_size, 1, cache_length)
            assert isinstance(key, MLAAttnKey)
            # The AMD sparse decode kernel splits the selected rows, not the
            # context, so the capture key must not move with the context.
            assert key.num_partitions == amd_partitions
            # The composite key folds in every leaf, the indexer's too.
            assert multi.resolve_attn_key(
                batch_size, 1, cache_length
            ) == multi.resolve_attn_key(batch_size, 1, 1)
        return
    # On SM100 the short-context FP8 policy can relax the split floor even
    # after the selection saturates, so only contexts past both share a key.
    # Capture probes stop there, so every verify width must share it too.
    saturated = max(sparse_indices_stride, _SPARSE_MLA_KEY_SATURATION)
    max_cache_length = max(4 * saturated, 100_000)
    probes = sparse.graph_capture_probe_cache_lengths(max_cache_length)
    assert [p for p in probes if p >= saturated] == [
        saturated,
        max_cache_length,
    ]
    for q_len in (1, 2, 4):
        key = sparse.resolve_attn_key(batch_size, q_len, saturated)
        assert isinstance(key, MLAAttnKey)
        multi_key = multi.resolve_attn_key(batch_size, q_len, saturated)
        for cache_length in (saturated + 1, 2 * saturated, max_cache_length):
            assert (
                sparse.resolve_attn_key(batch_size, q_len, cache_length) == key
            )
            assert (
                multi.resolve_attn_key(batch_size, q_len, cache_length)
                == multi_key
            )


# (batch_size, cache_length, num_partitions, kernel_variant) for a 128-head
# FP8 sparse decode over a 2048-row selection on SM100: the launch the
# sparse decode makes, which its graph ops do not override.
SPARSE_MLA_H128_CASES = [
    (1, 256, 1, 0),
    (1, 448, 1, 1),
    (1, 32768, 8, 0),
    (10, 1024, 4, 0),
    (10, 32768, 4, 1),
    (64, 32768, 1, 1),
]


@pytest.mark.skipif(
    accelerator_api() != "cuda", reason="SM100 sparse decode launch"
)
def test_sparse_mla_h128_dispatch_key_follows_the_launch(
    gpu_device_ref: DeviceRef,
) -> None:
    sparse = MLAKVCacheParams(
        dtype=DType.float8_e4m3fn,
        head_dim=576,
        num_layers=1,
        devices=[gpu_device_ref],
        num_q_heads=128,
        sparse_indices_stride=2048,
    )
    for (
        batch_size,
        cache_length,
        num_partitions,
        variant,
    ) in SPARSE_MLA_H128_CASES:
        key = sparse.resolve_attn_key(batch_size, 1, cache_length)
        assert key == MLAAttnKey(
            batch_size=batch_size,
            max_prompt_length=1,
            num_partitions=num_partitions,
            kernel_variant=variant,
        )
    # Past the selection the launch, and so the capture key, stops moving.
    multi = MultiKVCacheParams.from_params({"mla": sparse, "indexer": sparse})
    for batch_size in (1, 10, 64):
        assert multi.resolve_attn_key(
            batch_size, 1, 4096
        ) == multi.resolve_attn_key(batch_size, 1, 163_840)


# Columns: cache dtype, scales, index row stride, cache length, split count.
# Scaled FP8 and NVFP4 caches launch the dense heuristic of the top-k, with
# the relaxed split at caches of 1024 to 2048; NVFP4's index row is twice its
# top-k.
SPARSE_MLA_H128_FORM_CASES = [
    (
        DType.uint8,
        KVCacheQuantizationConfig(
            scale_dtype=DType.float8_e4m3fn,
            quantization_granularity=16,
            fp8_tail_dim=64,
        ),
        4096,
        1024,
        9,
    ),
    (
        DType.uint8,
        KVCacheQuantizationConfig(
            scale_dtype=DType.float8_e4m3fn,
            quantization_granularity=16,
            fp8_tail_dim=64,
        ),
        4096,
        32768,
        16,
    ),
    (
        DType.float8_e4m3fn,
        KVCacheQuantizationConfig(
            scale_dtype=DType.float32, quantization_granularity=64
        ),
        2048,
        1024,
        9,
    ),
    (
        DType.float8_e4m3fn,
        KVCacheQuantizationConfig(
            scale_dtype=DType.float32, quantization_granularity=64
        ),
        2048,
        32768,
        16,
    ),
]


@pytest.mark.skipif(
    accelerator_api() != "cuda", reason="SM100 sparse decode launch"
)
def test_sparse_mla_h128_dispatch_key_follows_the_cache_form(
    gpu_device_ref: DeviceRef,
) -> None:
    for (
        dtype,
        quant_config,
        stride,
        cache_length,
        num_partitions,
    ) in SPARSE_MLA_H128_FORM_CASES:
        params = MLAKVCacheParams(
            dtype=dtype,
            head_dim=576,
            num_layers=1,
            devices=[gpu_device_ref],
            num_q_heads=128,
            kvcache_quant_config=quant_config,
            sparse_indices_stride=stride,
        )
        assert params.resolve_attn_key(1, 1, cache_length) == MLAAttnKey(
            batch_size=1,
            max_prompt_length=1,
            num_partitions=num_partitions,
            kernel_variant=0,
        )


@pytest.mark.skipif(
    accelerator_api() != "cuda", reason="SM100 sparse decode launch"
)
def test_sparse_mla_h128_dispatch_key_follows_topk(
    gpu_device_ref: DeviceRef,
) -> None:
    def resolve(
        cache_length: int, page_size: int = 128, fixed_topk: bool = True
    ) -> MLAAttnKey:
        params = MLAKVCacheParams(
            dtype=DType.float8_e4m3fn,
            head_dim=576,
            num_layers=1,
            devices=[gpu_device_ref],
            num_q_heads=128,
            page_size=page_size,
            kvcache_quant_config=KVCacheQuantizationConfig(
                scale_dtype=DType.float32, quantization_granularity=64
            ),
            sparse_indices_stride=2048,
            sparse_fixed_topk=fixed_topk,
        )
        key = params.resolve_attn_key(1, 1, cache_length)
        assert isinstance(key, MLAAttnKey)
        return key

    # Fixed top-k takes the relaxed split, so the count tracks the cache's
    # pages (nine over the clamped 1025 keys); per-token top-k lengths keep
    # the page floor, so the count tracks the top-k's (sixteen over 2048).
    # The scaled-form split never reads the KV page size, so a 64-token page
    # moves no pick; the host pins in test_mla_decode_split_policy cover
    # where the page still reaches a record.
    assert resolve(1024).num_partitions == 9
    assert resolve(1024, fixed_topk=False).num_partitions == 16
    assert resolve(1024, page_size=64).num_partitions == 9


@pytest.mark.skipif(
    accelerator_api() != "cuda", reason="SM100 sparse decode launch"
)
def test_sparse_mla_indexer_key_follows_the_mla_cache_form(
    gpu_device_ref: DeviceRef,
) -> None:
    mla = MLAKVCacheParams(
        dtype=DType.float8_e4m3fn,
        head_dim=576,
        num_layers=1,
        devices=[gpu_device_ref],
        num_q_heads=128,
        sparse_indices_stride=2048,
    )
    # The indexer's FP8 cache carries per-token scales, which on its own
    # would key it on the scaled FP8 form.
    indexer = MLAKVCacheParams(
        dtype=DType.float8_e4m3fn,
        head_dim=128,
        num_layers=1,
        devices=[gpu_device_ref],
        num_q_heads=128,
        kvcache_quant_config=KVCacheQuantizationConfig(
            scale_dtype=DType.float32, quantization_granularity=128
        ),
        sparse_indices_stride=2048,
        sparse_kv_form=mla.resolved_sparse_kv_form,
    )
    assert mla.resolved_sparse_kv_form == MLASparseKVForm.PLAIN
    for batch_size, cache_length in ((1, 448), (1, 1024), (10, 32768)):
        assert indexer.resolve_attn_key(
            batch_size, 1, cache_length
        ) == mla.resolve_attn_key(batch_size, 1, cache_length)
