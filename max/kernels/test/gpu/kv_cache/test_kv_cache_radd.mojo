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

from gpu.host import DeviceContext
from utils import IndexList
from internal_utils import (
    HostNDBuffer,
    DeviceNDBuffer,
    initialize,
    InitializationType,
)
from kv_cache.types import PagedKVCacheCollection, KVCacheStaticParams
from math import ceildiv
from collections import Set
from random import random_ui64
from nn.kv_cache_ragged import generic_kv_cache_radd_dispatch
from buffer import Dim, DimList


fn test_kv_cache_radd[
    dtype: DType,
    num_heads: Int,
    head_dim: Int,
    page_size: Int,
    batch_size: Int,
](
    prompt_lens: IndexList[batch_size],
    cache_lens: IndexList[batch_size],
    num_active_loras: Int,
    ctx: DeviceContext,
) raises:
    alias num_layers = 2
    debug_assert(
        num_active_loras <= batch_size,
        "num_active_loras must be less than or equal to batch_size",
    )
    var input_row_offsets_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size + 1)
    )
    var cache_lengths_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size)
    )

    var input_row_offsets_slice_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](num_active_loras + 1)
    )
    var num_active_loras_slice_start = batch_size - num_active_loras
    var total_length = 0
    var total_slice_length = 0
    var max_full_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        input_row_offsets_host.tensor[i] = total_length
        cache_lengths_host.tensor[i] = cache_lens[i]
        max_full_context_length = max(
            max_full_context_length, cache_lens[i] + prompt_lens[i]
        )
        max_prompt_length = max(max_prompt_length, prompt_lens[i])

        if i >= num_active_loras_slice_start:
            input_row_offsets_slice_host.tensor[
                i - num_active_loras_slice_start
            ] = total_length
            total_slice_length += prompt_lens[i]

        total_length += prompt_lens[i]

    input_row_offsets_host.tensor[batch_size] = total_length
    input_row_offsets_slice_host.tensor[num_active_loras] = total_length

    num_paged_blocks = ceildiv(
        batch_size * max_full_context_length * 2, page_size
    )

    var cache_lengths_device = cache_lengths_host.copy_to_device(ctx)
    var input_row_offsets_slice_device = (
        input_row_offsets_slice_host.copy_to_device(ctx)
    )

    kv_block_paged_host = HostNDBuffer[dtype, 6](
        IndexList[6](
            num_paged_blocks,
            2,
            num_layers,
            page_size,
            num_heads,
            head_dim,
        )
    )
    initialize(kv_block_paged_host.tensor, InitializationType.one)
    paged_lut_host = HostNDBuffer[DType.uint32, 2](
        IndexList[2](batch_size, ceildiv(max_full_context_length, page_size))
    )
    paged_lut_set = Set[Int]()
    for bs in range(batch_size):
        seq_len = cache_lens[bs] + prompt_lens[bs]

        for block_idx in range(0, ceildiv(seq_len, page_size)):
            var randval = Int(random_ui64(0, num_paged_blocks - 1))
            while randval in paged_lut_set:
                randval = Int(random_ui64(0, num_paged_blocks - 1))

            paged_lut_set.add(randval)
            paged_lut_host.tensor[bs, block_idx] = randval

    kv_block_paged_device = kv_block_paged_host.copy_to_device(ctx)
    paged_lut_device = paged_lut_host.copy_to_device(ctx)

    var kv_collection_device = PagedKVCacheCollection[
        dtype,
        KVCacheStaticParams(
            num_heads=UInt(num_heads), head_size=UInt(head_dim)
        ),
        page_size,
    ](
        kv_block_paged_device.tensor,
        cache_lengths_device.tensor,
        paged_lut_device.tensor,
        max_prompt_length,
        max_full_context_length,
    )

    var a_host = HostNDBuffer[
        dtype, 2, DimList(Dim(), num_heads * head_dim * 2)
    ](IndexList[2](total_slice_length, num_heads * head_dim * 2))
    initialize(a_host.tensor, InitializationType.arange)
    var a_device = a_host.copy_to_device(ctx)

    var layer_idx = 1
    generic_kv_cache_radd_dispatch[target="gpu"](
        a_device.tensor,
        kv_collection_device,
        input_row_offsets_slice_device.tensor,
        num_active_loras_slice_start,
        layer_idx,
        ctx,
    )
    ctx.synchronize()
    ctx.enqueue_copy(
        kv_block_paged_host.tensor.data, kv_block_paged_device.buffer
    )
    ctx.enqueue_copy(a_host.tensor.data, a_device.buffer)

    ctx.synchronize()

    var kv_collection_host = PagedKVCacheCollection[
        dtype,
        KVCacheStaticParams(
            num_heads=UInt(num_heads), head_size=UInt(head_dim)
        ),
        page_size,
    ](
        kv_block_paged_host.tensor,
        cache_lengths_host.tensor,
        paged_lut_host.tensor,
        max_prompt_length,
        max_full_context_length,
    )

    var k_cache_host = kv_collection_host.get_key_cache(layer_idx)
    var v_cache_host = kv_collection_host.get_value_cache(layer_idx)

    # first check that we didn't augment previous cache entries
    for i in range(batch_size):
        for c in range(cache_lens[i]):
            for h in range(num_heads):
                for d in range(head_dim):
                    var k_val = k_cache_host.load[width=1](i, h, c, d)
                    var v_val = v_cache_host.load[width=1](i, h, c, d)
                    if k_val != 1:
                        raise Error(
                            "Mismatch in output for k, expected 1, got "
                            + String(k_val)
                            + " in k_cache at index "
                            + String(IndexList[4](i, c, h, d))
                        )
                    if v_val != 1:
                        raise Error(
                            "Mismatch in output for v, expected 1, got "
                            + String(v_val)
                            + " in v_cache at index "
                            + String(IndexList[4](i, c, h, d))
                        )

    # now check that we augmented the correct entries
    # the first elements in the batch should not be lora-augmented
    for i in range(batch_size - num_active_loras):
        for c in range(prompt_lens[i]):
            var actual_len = c + cache_lens[i]
            for h in range(num_heads):
                for d in range(head_dim):
                    var k_val = k_cache_host.load[width=1](i, h, actual_len, d)
                    var v_val = v_cache_host.load[width=1](i, h, actual_len, d)
                    if k_val != 1:
                        raise Error(
                            "Mismatch in output for k, expected 1, got "
                            + String(k_val)
                            + " in k_cache at index "
                            + String(IndexList[4](i, h, actual_len, d))
                        )
                    if v_val != 1:
                        raise Error(
                            "Mismatch in output for v, expected 1, got "
                            + String(v_val)
                            + " in v_cache at index "
                            + String(IndexList[4](i, h, actual_len, d))
                        )

    # now check that the lora-augmented entries are correct
    arange_counter = 0
    for i in range(batch_size - num_active_loras, batch_size):
        for c in range(prompt_lens[i]):
            var actual_len = c + cache_lens[i]
            for h in range(num_heads):
                for d in range(head_dim):
                    var k_val = k_cache_host.load[width=1](i, h, actual_len, d)
                    var expected_k_val = 1 + arange_counter
                    if k_val != expected_k_val:
                        raise Error(
                            "Mismatch in output for k, expected "
                            + String(expected_k_val)
                            + ", got "
                            + String(k_val)
                            + " in k_cache at index "
                            + String(IndexList[4](i, h, actual_len, d))
                        )
                    arange_counter += 1
            for h in range(num_heads):
                for d in range(head_dim):
                    var v_val = v_cache_host.load[width=1](i, h, actual_len, d)
                    var expected_v_val = 1 + arange_counter
                    if v_val != expected_v_val:
                        raise Error(
                            "Mismatch in output for v, expected "
                            + String(expected_v_val)
                            + ", got "
                            + String(v_val)
                            + " in v_cache at index "
                            + String(IndexList[4](i, h, actual_len, d))
                        )
                    arange_counter += 1

    # TODO(MAXPLAT-362) fix this and remove it.
    _ = paged_lut_host


fn main() raises:
    with DeviceContext() as ctx:
        test_kv_cache_radd[DType.float32, 8, 128, 128, 4,](
            IndexList[4](10, 20, 30, 40),
            IndexList[4](40, 30, 20, 10),
            2,
            ctx,
        )
        test_kv_cache_radd[DType.float32, 8, 128, 128, 4,](
            IndexList[4](10, 20, 30, 40),
            IndexList[4](40, 30, 20, 10),
            4,
            ctx,
        )
        test_kv_cache_radd[DType.float32, 8, 128, 128, 4,](
            IndexList[4](10, 20, 30, 40),
            IndexList[4](40, 30, 20, 10),
            0,
            ctx,
        )
        test_kv_cache_radd[DType.float32, 8, 128, 128, 1](
            IndexList[1](10),
            IndexList[1](40),
            1,
            ctx,
        )
