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
from max.driver import accelerator_api
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MLAKVCacheParams, MultiKVCacheParams
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
    dense = params(None)
    multi = MultiKVCacheParams.from_params({"mla": sparse, "indexer": sparse})
    for cache_length in SPARSE_MLA_CACHE_LENGTHS:
        key = sparse.resolve_attn_key(batch_size, 1, cache_length)
        assert isinstance(key, MLAAttnKey)
        if accelerator_api() == "hip":
            # The AMD sparse decode kernel splits the selected rows, not the
            # context, so the capture key must not move with the context.
            assert key.num_partitions == amd_partitions
            # The composite key folds in every leaf, the indexer's too.
            assert multi.resolve_attn_key(
                batch_size, 1, cache_length
            ) == multi.resolve_attn_key(batch_size, 1, 1)
        else:
            assert key == dense.resolve_attn_key(batch_size, 1, cache_length)
