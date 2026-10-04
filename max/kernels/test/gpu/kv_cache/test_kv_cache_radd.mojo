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

from std.math import ceildiv

from max.gpu.host import DeviceContext
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Coord, Idx, row_major
from layout._host_device_tile_tensor import HostDeviceTileTensor
from nn.kv_cache_ragged import generic_kv_cache_radd_dispatch

from std.utils import IndexList

from kv_cache_test_utils import PagedLookupTable


def test_kv_cache_radd[
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
    comptime num_layers = 2
    assert (
        num_active_loras <= batch_size
    ), "num_active_loras must be less than or equal to batch_size"
    var cache_lengths = HostDeviceTileTensor[.uint32](
        row_major(Int64(batch_size)), ctx
    )
    var input_row_offsets_slice = HostDeviceTileTensor[.uint32](
        row_major(Int64(num_active_loras + 1)), ctx
    )
    var cache_lengths_host = cache_lengths.host_tensor()
    var input_row_offsets_slice_host = input_row_offsets_slice.host_tensor()
    var num_active_loras_slice_start = batch_size - num_active_loras
    var running_total = 0
    var total_slice_length = 0
    var max_full_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lens[i])
        max_full_context_length = max(
            max_full_context_length, cache_lens[i] + prompt_lens[i]
        )
        max_prompt_length = max(max_prompt_length, prompt_lens[i])

        if i >= num_active_loras_slice_start:
            input_row_offsets_slice_host[
                i - num_active_loras_slice_start
            ] = UInt32(running_total)
            total_slice_length += prompt_lens[i]

        running_total += prompt_lens[i]

    input_row_offsets_slice_host[num_active_loras] = UInt32(running_total)
    cache_lengths.to_device()
    input_row_offsets_slice.to_device()

    var num_paged_blocks = ceildiv(
        batch_size * max_full_context_length * 2, page_size
    )

    var kv_block_paged = HostDeviceTileTensor[dtype](
        row_major(
            Int64(num_paged_blocks),
            Idx[2],
            Int64(num_layers),
            Idx[page_size],
            Idx[num_heads],
            Idx[head_dim],
        ),
        ctx,
    )
    _ = kv_block_paged.host_tensor().fill(1)
    kv_block_paged.to_device()

    var paged_lut = PagedLookupTable[page_size].build(
        prompt_lens, cache_lens, max_full_context_length, num_paged_blocks, ctx
    )

    comptime Collection = PagedKVCacheCollection[
        dtype,
        KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var kv_collection_device = Collection(
        kv_block_paged.device_tensor().as_unsafe_any_origin(),
        cache_lengths.device_tensor().as_imm().as_unsafe_any_origin(),
        paged_lut.device_tile_tensor(),
        UInt32(max_prompt_length),
        UInt32(max_full_context_length),
    )

    var a = HostDeviceTileTensor[dtype](
        row_major(Int64(total_slice_length), Idx[num_heads * head_dim * 2]),
        ctx,
    )
    var a_host = a.host_tensor()
    for i in range(a_host.num_elements()):
        a_host.unsafe_ptr().unsafe_store(i, Scalar[dtype](i))
    a.to_device()

    var layer_idx = 1
    generic_kv_cache_radd_dispatch[target="gpu"](
        a.device_tensor(),
        kv_collection_device,
        input_row_offsets_slice.device_tensor(),
        UInt32(num_active_loras_slice_start),
        UInt32(layer_idx),
        ctx,
    )
    ctx.synchronize()
    kv_block_paged.to_host()

    var kv_collection_host = Collection(
        kv_block_paged.host_tensor().as_unsafe_any_origin(),
        cache_lengths.host_tensor().as_imm().as_unsafe_any_origin(),
        paged_lut.host_tile_tensor(),
        UInt32(max_prompt_length),
        UInt32(max_full_context_length),
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
    var arange_counter = 0
    for i in range(batch_size - num_active_loras, batch_size):
        for c in range(prompt_lens[i]):
            var actual_len = c + cache_lens[i]
            for h in range(num_heads):
                for d in range(head_dim):
                    var k_val = k_cache_host.load[width=1](i, h, actual_len, d)
                    var expected_k_val = 1 + arange_counter
                    if k_val != Scalar[dtype](expected_k_val):
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
                    if v_val != Scalar[dtype](expected_v_val):
                        raise Error(
                            "Mismatch in output for v, expected "
                            + String(expected_v_val)
                            + ", got "
                            + String(v_val)
                            + " in v_cache at index "
                            + String(IndexList[4](i, h, actual_len, d))
                        )
                    arange_counter += 1

    # The collections hold untracked views of these buffers.
    _ = kv_block_paged^
    _ = cache_lengths^
    _ = paged_lut^


def main() raises:
    with DeviceContext() as ctx:
        test_kv_cache_radd[.float32, 8, 128, 128](
            IndexList[4](10, 20, 30, 40),
            IndexList[4](40, 30, 20, 10),
            2,
            ctx,
        )
        test_kv_cache_radd[.float32, 8, 128, 128](
            IndexList[4](10, 20, 30, 40),
            IndexList[4](40, 30, 20, 10),
            4,
            ctx,
        )
        test_kv_cache_radd[.float32, 8, 128, 128](
            IndexList[4](10, 20, 30, 40),
            IndexList[4](40, 30, 20, 10),
            0,
            ctx,
        )
        test_kv_cache_radd[.float32, 8, 128, 128](
            IndexList[1](10),
            IndexList[1](40),
            1,
            ctx,
        )
