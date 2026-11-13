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
"""Tests for NullKVCache, the cache a pipeline measures its warmups against."""

from __future__ import annotations

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheInputsPerDevice,
    MHAKVCacheParams,
)
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.kv_cache import JengaKVCacheManager, NullKVCache
from max.pipelines.request.base import RequestID


def _params(page_size: int = 128) -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.float32,
        num_layers=1,
        n_kv_heads=2,
        head_dim=1,
        page_size=page_size,
        devices=[DeviceRef.CPU()],
    )


def _ctx(num_tokens: int) -> TextContext:
    return TextContext(
        request_id=RequestID(),
        max_length=1 << 20,
        tokens=TokenBuffer(np.zeros(num_tokens, dtype=np.int64)),
    )


def _per_device(
    inputs: KVCacheInputs[Buffer, Buffer],
) -> KVCacheInputsPerDevice[Buffer, Buffer]:
    assert isinstance(inputs, tuple) and inputs
    device = inputs[0]
    assert isinstance(device, KVCacheInputsPerDevice)
    return device


def test_any_length_runs_on_the_null_page() -> None:
    cache = NullKVCache(params=_params(), max_batch_size=4, max_seq_len=1 << 16)
    long, short = _ctx(40_000), _ctx(3)
    for ctx in (long, short):
        cache.alloc_dummy(ctx)
    inputs = _per_device(cache.runtime_inputs([[long, short]]))
    lut = inputs.lookup_table.to_numpy()
    # 40,000 tokens reach 313 pages, every one the null page.
    assert lut.shape[0] == 2 and lut.shape[1] >= 313
    assert not lut.any()
    assert inputs.cache_lengths.to_numpy().tolist() == [0, 0]
    assert cache.contains(long)
    cache.release(long)
    assert not cache.contains(long)
    assert cache.block_count().free == 1


def test_inputs_match_the_real_cache_for_the_same_batch() -> None:
    params = _params(page_size=128)
    real = JengaKVCacheManager.create(
        params=params,
        available_bytes=10 * params.bytes_per_block,
        max_batch_size=8,
        max_seq_len=256,
    )
    null = NullKVCache(params=params, max_batch_size=8, max_seq_len=256)
    a, b = _ctx(3), _ctx(2)
    for ctx in (a, b):
        real.alloc_dummy(ctx)
        null.alloc_dummy(ctx)
    got = _per_device(null.runtime_inputs([[a, b]]))
    want = _per_device(real.runtime_inputs([[a, b]]))
    # Same staged inputs, same shapes: a graph cannot tell the two apart.
    assert got.lookup_table.shape == want.lookup_table.shape
    np.testing.assert_array_equal(
        got.lookup_table.to_numpy(), want.lookup_table.to_numpy()
    )
    np.testing.assert_array_equal(
        got.cache_lengths.to_numpy(), want.cache_lengths.to_numpy()
    )
