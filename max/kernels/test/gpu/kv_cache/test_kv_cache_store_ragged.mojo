# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from collections import Optional, Set
from math import ceildiv
from random import random_ui64, seed

from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from layout import *
from layout._utils import ManagedLayoutTensor, UNKNOWN_VALUE
from memory import memcpy, memset_zero
from nn.kv_cache_ragged import kv_cache_store_ragged
from testing import assert_equal, assert_almost_equal

from utils import IndexList

alias kv_params_test = KVCacheStaticParams(num_heads=4, head_size=64)
alias dtype = DType.float32


fn test_kv_cache_store_ragged_basic(ctx: DeviceContext) raises:
    alias dtype = DType.float32
    alias page_size = 128
    alias num_kv_heads = 2
    alias kv_params = KVCacheStaticParams(num_heads=num_kv_heads, head_size=64)
    alias num_layers = 2
    alias batch_size = 3
    var valid_lengths = List[Int](100, 200, 300)
    var cache_lengths = List[Int](100, 200, 300)

    debug_assert(
        len(valid_lengths) == len(cache_lengths),
        "expected valid_lengths and cache_lengths size to be equal",
    )

    alias cache_lengths_layout = Layout.row_major(UNKNOWN_VALUE)
    var cache_lengths_runtime_layout = RuntimeLayout[
        cache_lengths_layout
    ].row_major(IndexList[1](batch_size))
    var cache_lengths_managed = ManagedLayoutTensor[
        DType.uint32, cache_lengths_layout
    ](cache_lengths_runtime_layout, ctx)
    var cache_lengths_tensor = cache_lengths_managed.tensor()

    alias input_row_offsets_layout = Layout.row_major(UNKNOWN_VALUE)
    var runtime_layout = RuntimeLayout[input_row_offsets_layout].row_major(
        IndexList[1](batch_size + 1)
    )
    var input_row_offsets_managed = ManagedLayoutTensor[
        DType.uint32, input_row_offsets_layout
    ](runtime_layout, ctx)

    var total_length = 0
    var max_full_context_length = 0
    var max_prompt_length = 0
    input_row_offsets_buf = input_row_offsets_managed.buffer()
    for i in range(batch_size):
        input_row_offsets_buf[i] = total_length
        cache_lengths_tensor[i] = cache_lengths[i]
        max_full_context_length = max(
            max_full_context_length, cache_lengths[i] + valid_lengths[i]
        )
        max_prompt_length = max(max_prompt_length, valid_lengths[i])
        total_length += valid_lengths[i]
    input_row_offsets_buf[batch_size] = total_length

    var num_paged_blocks = (
        ceildiv(max_full_context_length, page_size) * batch_size
    )

    var kv_block_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    alias kv_block_layout = Layout.row_major(
        UNKNOWN_VALUE,
        UNKNOWN_VALUE,
        UNKNOWN_VALUE,
        UNKNOWN_VALUE,
        UNKNOWN_VALUE,
        UNKNOWN_VALUE,
    )
    var kv_block_runtime_layout = RuntimeLayout[kv_block_layout].row_major(
        kv_block_shape
    )
    var kv_block_managed = ManagedLayoutTensor[dtype, kv_block_layout](
        kv_block_runtime_layout, ctx
    )
    var kv_block_tensor = kv_block_managed.tensor()

    var paged_lut_shape = IndexList[2](
        batch_size, ceildiv(max_full_context_length, page_size)
    )
    alias paged_lut_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    var paged_lut_runtime_layout = RuntimeLayout[paged_lut_layout].row_major(
        paged_lut_shape
    )
    var paged_lut_managed = ManagedLayoutTensor[DType.uint32, paged_lut_layout](
        paged_lut_runtime_layout, ctx
    )
    var paged_lut_tensor = paged_lut_managed.tensor()
    var paged_lut_set = Set[Int]()
    for bs in range(batch_size):
        var seq_len = cache_lengths[bs] + valid_lengths[bs]

        for block_idx in range(0, ceildiv(seq_len, page_size)):
            var randval = Int(random_ui64(0, num_paged_blocks - 1))
            while randval in paged_lut_set:
                randval = Int(random_ui64(0, num_paged_blocks - 1))

            paged_lut_set.add(randval)
            paged_lut_tensor[bs, block_idx] = randval

    var kv_collection_paged_device = PagedKVCacheCollection[
        dtype, kv_params, page_size
    ](
        rebind[NDBuffer[dtype, 6, MutableAnyOrigin]](
            kv_block_managed.device_buffer()
        ),
        rebind[NDBuffer[DType.uint32, 1, MutableAnyOrigin]](
            cache_lengths_managed.device_buffer()
        ),
        rebind[NDBuffer[DType.uint32, 2, MutableAnyOrigin]](
            paged_lut_managed.device_buffer()
        ),
        max_prompt_length,
        max_full_context_length,
    )

    var q_shape = IndexList[3](total_length, num_kv_heads, kv_params.head_size)
    alias q_layout = Layout.row_major(
        UNKNOWN_VALUE, num_kv_heads, kv_params.head_size
    )
    var q_runtime_layout = RuntimeLayout[q_layout].row_major(q_shape)
    var q_managed = ManagedLayoutTensor[dtype, q_layout](q_runtime_layout, ctx)
    var q_tensor = q_managed.tensor()

    # Fill input data for testing
    var current_offset = 0
    for batch_idx in range(batch_size):
        var seq_len = valid_lengths[batch_idx]
        for token_idx in range(seq_len):
            for head_idx in range(num_kv_heads):
                for head_dim_idx in range(kv_params.head_size):
                    # Calculate expected value
                    var global_token_idx = current_offset + token_idx
                    var expected_linear_idx = (
                        global_token_idx * num_kv_heads * kv_params.head_size
                        + head_idx * kv_params.head_size
                        + head_dim_idx
                    )
                    q_tensor[
                        global_token_idx, head_idx, head_dim_idx
                    ] = Float32(expected_linear_idx)
        current_offset += seq_len

    var q_device_tensor = q_managed.device_tensor()

    @parameter
    @always_inline
    @__copy_capture(q_device_tensor)
    fn input_fn[
        width: Int, alignment: Int
    ](idx: IndexList[3]) -> SIMD[dtype, width]:
        return q_device_tensor.load[width](idx)

    var k_cache_device = kv_collection_paged_device.get_key_cache(0)
    kv_cache_store_ragged[input_fn=input_fn, target="gpu"](
        k_cache_device,
        q_shape,
        input_row_offsets_managed.device_tensor(),
        ctx,
    )
    ctx.synchronize()

    var kv_collection_paged_host = PagedKVCacheCollection[
        dtype, kv_params, page_size
    ](
        rebind[NDBuffer[dtype, 6, MutableAnyOrigin]](kv_block_managed.buffer()),
        rebind[NDBuffer[DType.uint32, 1, MutableAnyOrigin]](
            cache_lengths_managed.buffer()
        ),
        rebind[NDBuffer[DType.uint32, 2, MutableAnyOrigin]](
            paged_lut_managed.buffer()
        ),
        max_prompt_length,
        max_full_context_length,
    )
    var k_cache_host = kv_collection_paged_host.get_key_cache(0)

    # Verify the data was stored correctly
    current_offset = 0
    for batch_idx in range(batch_size):
        var seq_len = valid_lengths[batch_idx]
        for token_idx in range(seq_len):
            for head_idx in range(num_kv_heads):
                for head_dim_idx in range(kv_params.head_size):
                    # Calculate expected value
                    var global_token_idx = current_offset + token_idx
                    var expected_linear_idx = (
                        global_token_idx * num_kv_heads * kv_params.head_size
                        + head_idx * kv_params.head_size
                        + head_dim_idx
                    )
                    var expected_value = Float32(expected_linear_idx)

                    # Get actual value from cache
                    var cache_token_idx = token_idx + cache_lengths[batch_idx]
                    var actual_value = k_cache_host.load[width=1](
                        batch_idx,
                        UInt(head_idx),
                        cache_token_idx,
                        UInt(head_dim_idx),
                    )

                    # Verify the values match
                    assert_almost_equal(
                        actual_value,
                        expected_value,
                        rtol=1e-5,
                        atol=1e-6,
                        msg="Mismatch at batch="
                        + String(batch_idx)
                        + ", token="
                        + String(token_idx)
                        + ", head="
                        + String(head_idx)
                        + ", head_dim="
                        + String(head_dim_idx)
                        + ": expected "
                        + String(expected_value)
                        + ", got "
                        + String(actual_value),
                    )
        current_offset += seq_len
    _ = kv_block_managed^
    _ = paged_lut_managed^
    _ = cache_lengths_managed^
    _ = input_row_offsets_managed^


def main():
    seed(42)  # Set seed for reproducible tests

    with DeviceContext() as ctx:
        test_kv_cache_store_ragged_basic(ctx)
