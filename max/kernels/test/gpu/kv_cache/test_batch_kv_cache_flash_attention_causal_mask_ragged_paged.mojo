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

from collections import Set
from math import ceildiv, rsqrt
from random import random_ui64, seed

from gpu.host import DeviceContext
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from memory import memcpy, memset_zero
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from nn.mha import flash_attention
from nn.mha_mask import CausalMask
from nn.mha_score_mod import IdentityScoreMod
from testing import assert_almost_equal, assert_equal
from sys import has_nvidia_gpu_accelerator

from utils import IndexList

comptime kv_params_llama3 = KVCacheStaticParams(num_heads=8, head_size=128)
comptime kv_params_llama3_1b = KVCacheStaticParams(num_heads=8, head_size=64)
comptime llama_num_q_heads = 32


def execute_ragged_flash_attention[
    num_q_heads: Int, dtype: DType, kv_params: KVCacheStaticParams
](
    valid_lengths: List[Int],
    cache_lengths: List[Int],
    num_layers: Int,
    layer_idx: Int,
    ctx: DeviceContext,
):
    comptime page_size = 512

    var batch_size = len(valid_lengths)
    debug_assert(
        len(valid_lengths) == len(cache_lengths),
        "expected valid_lengths and cache_lengths size to be equal",
    )

    # Compute dimensions
    var total_length = 0
    var max_full_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        max_full_context_length = max(
            max_full_context_length, cache_lengths[i] + valid_lengths[i]
        )
        max_prompt_length = max(max_prompt_length, valid_lengths[i])
        total_length += valid_lengths[i]

    # Define layouts
    comptime row_offsets_layout = Layout(UNKNOWN_VALUE)
    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    comptime q_ragged_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, Int(kv_params.head_size)
    )
    comptime output_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, Int(kv_params.head_size)
    )
    comptime lookup_table_layout = Layout(UNKNOWN_VALUE)
    comptime paged_lut_layout = Layout.row_major[2]()
    comptime kv_block_6d_layout = Layout.row_major[6]()

    # Create shapes
    var row_offsets_shape = IndexList[1](batch_size + 1)
    var cache_lengths_shape = IndexList[1](batch_size)
    var q_ragged_shape = IndexList[3](
        total_length, num_q_heads, Int(kv_params.head_size)
    )
    var output_shape = IndexList[3](
        total_length, num_q_heads, Int(kv_params.head_size)
    )

    # Create runtime layouts
    var row_offsets_runtime_layout = RuntimeLayout[
        row_offsets_layout
    ].row_major(row_offsets_shape)
    var cache_lengths_runtime_layout = RuntimeLayout[
        cache_lengths_layout
    ].row_major(cache_lengths_shape)
    var q_ragged_runtime_layout = RuntimeLayout[q_ragged_layout].row_major(
        q_ragged_shape
    )
    var output_runtime_layout = RuntimeLayout[output_layout].row_major(
        output_shape
    )
    var lookup_table_runtime_layout = RuntimeLayout[
        lookup_table_layout
    ].row_major(cache_lengths_shape)

    # Create device buffers
    var input_row_offsets_device = ctx.enqueue_create_buffer[DType.uint32](
        row_offsets_shape.flattened_length()
    )
    var cache_lengths_device = ctx.enqueue_create_buffer[DType.uint32](
        cache_lengths_shape.flattened_length()
    )
    var q_ragged_device = ctx.enqueue_create_buffer[dtype](
        q_ragged_shape.flattened_length()
    )
    var test_output_device = ctx.enqueue_create_buffer[dtype](
        output_shape.flattened_length()
    )
    var ref_output_device = ctx.enqueue_create_buffer[dtype](
        output_shape.flattened_length()
    )

    # Host pointers for data that needs to persist for verification
    var input_row_offsets_host_ptr = UnsafePointer[UInt32].alloc(
        row_offsets_shape.flattened_length()
    )
    var test_output_host_ptr = UnsafePointer[Scalar[dtype]].alloc(
        output_shape.flattened_length()
    )
    var ref_output_host_ptr = UnsafePointer[Scalar[dtype]].alloc(
        output_shape.flattened_length()
    )

    # Initialize input_row_offsets and cache_lengths
    var running_offset: UInt32 = 0
    with cache_lengths_device.map_to_host() as cache_lengths_host_ptr:
        for i in range(batch_size):
            input_row_offsets_host_ptr[i] = running_offset
            cache_lengths_host_ptr[i] = UInt32(cache_lengths[i])
            running_offset += UInt32(valid_lengths[i])
        input_row_offsets_host_ptr[batch_size] = running_offset
    ctx.enqueue_copy(input_row_offsets_device, input_row_offsets_host_ptr)

    # Initialize q_ragged with random data
    with q_ragged_device.map_to_host() as q_ragged_host_ptr:
        var q_ragged_tensor = LayoutTensor[dtype, q_ragged_layout](
            q_ragged_host_ptr, q_ragged_runtime_layout
        )
        random(q_ragged_tensor)

    var num_continuous_blocks = batch_size + 2
    var num_paged_blocks = (
        ceildiv(max_full_context_length, page_size) * batch_size
    )

    # KV block shapes
    var kv_block_continuous_shape = IndexList[6](
        num_continuous_blocks,
        2,
        num_layers,
        max_full_context_length,
        Int(kv_params.num_heads),
        Int(kv_params.head_size),
    )
    var kv_block_paged_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        Int(kv_params.num_heads),
        Int(kv_params.head_size),
    )
    var paged_lut_shape = IndexList[2](
        batch_size, ceildiv(max_full_context_length, page_size)
    )

    # KV block runtime layouts
    var kv_block_continuous_runtime_layout = RuntimeLayout[
        kv_block_6d_layout
    ].row_major(kv_block_continuous_shape)
    var kv_block_paged_runtime_layout = RuntimeLayout[
        kv_block_6d_layout
    ].row_major(kv_block_paged_shape)
    var paged_lut_runtime_layout = RuntimeLayout[paged_lut_layout].row_major(
        paged_lut_shape
    )

    # Host pointers for KV blocks (need to persist for memcpy operations)
    var kv_block_continuous_host_ptr = UnsafePointer[Scalar[dtype]].alloc(
        kv_block_continuous_shape.flattened_length()
    )
    var kv_block_paged_host_ptr = UnsafePointer[Scalar[dtype]].alloc(
        kv_block_paged_shape.flattened_length()
    )
    var lookup_table_host_ptr = UnsafePointer[UInt32].alloc(batch_size)
    var paged_lut_host_ptr = UnsafePointer[UInt32].alloc(
        paged_lut_shape.flattened_length()
    )

    # Initialize kv_block_continuous with random data
    var kv_block_continuous_tensor = LayoutTensor[dtype, kv_block_6d_layout](
        kv_block_continuous_host_ptr, kv_block_continuous_runtime_layout
    )
    random(kv_block_continuous_tensor)

    # Create device buffers for KV blocks
    var kv_block_continuous_device = ctx.enqueue_create_buffer[dtype](
        kv_block_continuous_shape.flattened_length()
    )
    ctx.enqueue_copy(kv_block_continuous_device, kv_block_continuous_host_ptr)

    var lookup_table_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )

    # Hacky way to select random blocks for continuous batching
    var block_idx_set = Set[Int]()
    var idx = 0
    while idx < batch_size:
        var randval = Int(random_ui64(0, UInt64(num_continuous_blocks - 1)))
        if randval in block_idx_set:
            continue

        block_idx_set.add(randval)
        lookup_table_host_ptr[idx] = UInt32(randval)
        idx += 1
    ctx.enqueue_copy(lookup_table_device, lookup_table_host_ptr)

    # Create LayoutTensors for KV collection
    var kv_block_continuous_lt = LayoutTensor[
        dtype, kv_block_6d_layout, MutAnyOrigin
    ](kv_block_continuous_device, kv_block_continuous_runtime_layout)
    var cache_lengths_lt = LayoutTensor[
        DType.uint32, cache_lengths_layout, ImmutAnyOrigin
    ](cache_lengths_device, cache_lengths_runtime_layout)
    var lookup_table_lt = LayoutTensor[
        DType.uint32, lookup_table_layout, ImmutAnyOrigin
    ](lookup_table_device, lookup_table_runtime_layout)

    kv_collection_continuous_device = ContinuousBatchingKVCacheCollection[
        dtype, kv_params
    ](
        kv_block_continuous_lt,
        cache_lengths_lt,
        lookup_table_lt,
        UInt32(max_prompt_length),
        UInt32(max_full_context_length),
    )

    # Initialize paged KV cache by copying from continuous
    var kv_block_paged_tensor = LayoutTensor[dtype, kv_block_6d_layout](
        kv_block_paged_host_ptr, kv_block_paged_runtime_layout
    )
    var paged_lut_tensor = LayoutTensor[DType.uint32, paged_lut_layout](
        paged_lut_host_ptr, paged_lut_runtime_layout
    )

    paged_lut_set = Set[Int]()
    for bs in range(batch_size):
        seq_len = cache_lengths[bs] + valid_lengths[bs]
        continuous_idx = Int(lookup_table_host_ptr[bs])

        for block_idx in range(0, ceildiv(seq_len, page_size)):
            var randval = Int(random_ui64(0, UInt64(num_paged_blocks - 1)))
            while randval in paged_lut_set:
                randval = Int(random_ui64(0, UInt64(num_paged_blocks - 1)))

            paged_lut_set.add(randval)
            paged_lut_host_ptr[bs * paged_lut_shape[1] + block_idx] = UInt32(
                randval
            )
            block_sz = min(page_size, seq_len - block_idx * page_size)

            for kv_idx in range(2):
                # Calculate offsets manually for the 6D tensors
                var paged_offset = (
                    randval
                    * kv_block_paged_shape[1]
                    * kv_block_paged_shape[2]
                    * kv_block_paged_shape[3]
                    * kv_block_paged_shape[4]
                    * kv_block_paged_shape[5]
                    + kv_idx
                    * kv_block_paged_shape[2]
                    * kv_block_paged_shape[3]
                    * kv_block_paged_shape[4]
                    * kv_block_paged_shape[5]
                    + layer_idx
                    * kv_block_paged_shape[3]
                    * kv_block_paged_shape[4]
                    * kv_block_paged_shape[5]
                )
                var continuous_offset = (
                    continuous_idx
                    * kv_block_continuous_shape[1]
                    * kv_block_continuous_shape[2]
                    * kv_block_continuous_shape[3]
                    * kv_block_continuous_shape[4]
                    * kv_block_continuous_shape[5]
                    + kv_idx
                    * kv_block_continuous_shape[2]
                    * kv_block_continuous_shape[3]
                    * kv_block_continuous_shape[4]
                    * kv_block_continuous_shape[5]
                    + layer_idx
                    * kv_block_continuous_shape[3]
                    * kv_block_continuous_shape[4]
                    * kv_block_continuous_shape[5]
                    + block_idx
                    * page_size
                    * kv_block_continuous_shape[4]
                    * kv_block_continuous_shape[5]
                )
                var n_cpy = block_sz * Int(
                    kv_params.num_heads * kv_params.head_size
                )
                memcpy(
                    dest=kv_block_paged_host_ptr + paged_offset,
                    src=kv_block_continuous_host_ptr + continuous_offset,
                    count=n_cpy,
                )
                if block_sz < page_size:
                    memset_zero(
                        kv_block_paged_host_ptr + paged_offset + n_cpy,
                        (page_size - block_sz)
                        * Int(kv_params.num_heads * kv_params.head_size),
                    )

    # Create device buffers and copy paged data
    var kv_block_paged_device = ctx.enqueue_create_buffer[dtype](
        kv_block_paged_shape.flattened_length()
    )
    var paged_lut_device = ctx.enqueue_create_buffer[DType.uint32](
        paged_lut_shape.flattened_length()
    )
    ctx.enqueue_copy(kv_block_paged_device, kv_block_paged_host_ptr)
    ctx.enqueue_copy(paged_lut_device, paged_lut_host_ptr)

    # Create LayoutTensors for paged KV collection
    var kv_block_paged_lt = LayoutTensor[
        dtype, kv_block_6d_layout, MutAnyOrigin
    ](kv_block_paged_device, kv_block_paged_runtime_layout)
    var paged_lut_lt = LayoutTensor[
        DType.uint32, paged_lut_layout, ImmutAnyOrigin
    ](paged_lut_device, paged_lut_runtime_layout)

    kv_collection_paged_device = PagedKVCacheCollection[
        dtype, kv_params, page_size
    ](
        kv_block_paged_lt,
        cache_lengths_lt,
        paged_lut_lt,
        UInt32(max_prompt_length),
        UInt32(max_full_context_length),
    )

    # Create LayoutTensors for flash attention calls
    var q_ragged_lt = LayoutTensor[dtype, q_ragged_layout, MutAnyOrigin](
        q_ragged_device, q_ragged_runtime_layout
    )
    var ref_output_lt = LayoutTensor[dtype, output_layout, MutAnyOrigin](
        ref_output_device, output_runtime_layout
    )
    var test_output_lt = LayoutTensor[dtype, output_layout, MutAnyOrigin](
        test_output_device, output_runtime_layout
    )
    var input_row_offsets_lt = LayoutTensor[
        DType.uint32, row_offsets_layout, MutAnyOrigin
    ](input_row_offsets_device, row_offsets_runtime_layout)

    # continuous execution
    flash_attention[ragged=True](
        ref_output_lt,
        q_ragged_lt,
        kv_collection_continuous_device.get_key_cache(layer_idx),
        kv_collection_continuous_device.get_value_cache(layer_idx),
        CausalMask(),
        IdentityScoreMod(),
        input_row_offsets_lt,
        rsqrt(Float32(kv_params.head_size)),
        ctx,
    )

    # paged execution
    flash_attention[ragged=True](
        test_output_lt,
        q_ragged_lt,
        kv_collection_paged_device.get_key_cache(layer_idx),
        kv_collection_paged_device.get_value_cache(layer_idx),
        CausalMask(),
        IdentityScoreMod(),
        input_row_offsets_lt,
        rsqrt(Float32(kv_params.head_size)),
        ctx,
    )
    ctx.enqueue_copy(test_output_host_ptr, test_output_device)
    ctx.enqueue_copy(ref_output_host_ptr, ref_output_device)
    ctx.synchronize()

    # Create host tensors for verification
    var ref_out = LayoutTensor[dtype, output_layout](
        ref_output_host_ptr, output_runtime_layout
    )
    var test_out = LayoutTensor[dtype, output_layout](
        test_output_host_ptr, output_runtime_layout
    )

    for bs in range(batch_size):
        prompt_len = valid_lengths[bs]
        ragged_offset = Int(input_row_offsets_host_ptr[bs])
        for s in range(prompt_len):
            for h in range(num_q_heads):
                for hd in range(kv_params.head_size):
                    try:
                        assert_almost_equal(
                            ref_out[ragged_offset + s, h, Int(hd)],
                            test_out[ragged_offset + s, h, Int(hd)],
                            atol=1e-2,
                        )
                    except e:
                        print(
                            "MISMATCH:",
                            bs,
                            s,
                            h,
                            hd,
                            ref_out[ragged_offset + s, h, Int(hd)],
                            test_out[ragged_offset + s, h, Int(hd)],
                        )
                        raise e^

    # check reproducibility
    for repeat in range(16):
        flash_attention[ragged=True](
            ref_output_lt,
            q_ragged_lt,
            kv_collection_paged_device.get_key_cache(layer_idx),
            kv_collection_paged_device.get_value_cache(layer_idx),
            CausalMask(),
            IdentityScoreMod(),
            input_row_offsets_lt,
            rsqrt(Float32(kv_params.head_size)),
            ctx,
        )
        ctx.enqueue_copy(ref_output_host_ptr, ref_output_device)
        ctx.synchronize()
        for bs in range(batch_size):
            prompt_len = valid_lengths[bs]
            ragged_offset = Int(input_row_offsets_host_ptr[bs])
            for s in range(prompt_len):
                for h in range(num_q_heads):
                    for d in range(kv_params.head_size):
                        rep = ref_out[ragged_offset + s, h, Int(d)]
                        orig = test_out[ragged_offset + s, h, Int(d)]
                        if rep != orig:
                            print("repeat s h d =", repeat, s, h, d)
                        assert_equal(rep, orig)
                        ref_out[ragged_offset + s, h, Int(d)] = 123.4567

    # Free host pointers
    input_row_offsets_host_ptr.free()
    test_output_host_ptr.free()
    ref_output_host_ptr.free()
    kv_block_continuous_host_ptr.free()
    kv_block_paged_host_ptr.free()
    lookup_table_host_ptr.free()
    paged_lut_host_ptr.free()

    # Free device buffers
    _ = q_ragged_device^
    _ = kv_block_continuous_device^
    _ = kv_block_paged_device^
    _ = lookup_table_device^
    _ = ref_output_device^
    _ = test_output_device^
    _ = cache_lengths_device^
    _ = paged_lut_device^
    _ = input_row_offsets_device^


def execute_flash_attention_suite(ctx: DeviceContext):
    comptime types = (DType.float32, DType.bfloat16)

    for bs in [1, 4, 16]:
        comptime for type_idx in range(len(types)):
            comptime type = types[type_idx]
            if bs == 16 and type == DType.float32:
                # This fails for the MI300X
                continue
            ce_cache_sizes = List[Int]()
            ce_seq_lens = List[Int]()
            tg_cache_sizes = List[Int]()
            tg_seq_lens = List[Int]()
            for _ in range(bs):
                tg_seq_lens.append(1)
                tg_cache_sizes.append(Int(random_ui64(1, 1024)))
                ce_seq_lens.append(Int(random_ui64(1, 1024)))
                ce_cache_sizes.append(0)

            print("CE", bs, type)
            execute_ragged_flash_attention[
                llama_num_q_heads, type, kv_params_llama3
            ](ce_seq_lens, ce_cache_sizes, 2, 1, ctx)
            execute_ragged_flash_attention[
                llama_num_q_heads, type, kv_params_llama3_1b
            ](ce_seq_lens, ce_cache_sizes, 2, 1, ctx)

            print("TG", bs, type)
            execute_ragged_flash_attention[
                llama_num_q_heads, type, kv_params_llama3
            ](tg_seq_lens, tg_cache_sizes, 2, 0, ctx)

            comptime if has_nvidia_gpu_accelerator():
                print("TG", bs, type, "q_heads//kv_heads = 16//1")
                execute_ragged_flash_attention[
                    16,
                    type,
                    KVCacheStaticParams(num_heads=1, head_size=128),
                ](tg_seq_lens, tg_cache_sizes, 2, 0, ctx)
                print("TG", bs, type, "q_heads//kv_heads = 32//2")
                execute_ragged_flash_attention[
                    32,
                    type,
                    KVCacheStaticParams(num_heads=2, head_size=128),
                ](tg_seq_lens, tg_cache_sizes, 2, 0, ctx)

    # edge cases
    print("CE", 1, DType.bfloat16)
    for len in [2, 27]:
        var short_ce_seq_len = [len]
        var short_ce_cache_size = [0]
        execute_ragged_flash_attention[
            llama_num_q_heads, DType.bfloat16, kv_params_llama3
        ](short_ce_seq_len, short_ce_cache_size, 2, 1, ctx)

    print("TG", 2, DType.bfloat16)
    tg_seq_lens = [1, 1]
    tg_variable_cache_lens = [1024, 11]
    execute_ragged_flash_attention[
        llama_num_q_heads, DType.bfloat16, kv_params_llama3
    ](tg_seq_lens, tg_variable_cache_lens, 2, 0, ctx)


def main():
    seed(42)
    with DeviceContext() as ctx:
        execute_flash_attention_suite(ctx)
