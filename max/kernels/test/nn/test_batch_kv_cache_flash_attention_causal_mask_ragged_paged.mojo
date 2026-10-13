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

from std.collections import Set
from std.math import ceildiv, inf, rsqrt
from std.random import random_ui64, seed

from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from layout import Coord, Idx, TileTensor, row_major
from layout._fillers import random
from std.memory import unsafe_memcpy
from nn.attention.cpu.mha import flash_attention_kv_cache
from nn.attention.mha_mask import CausalMask
from std.testing import assert_almost_equal, assert_equal, assert_true
from std.sys import size_of

from std.utils import IndexList

comptime kv_params_replit = KVCacheStaticParams(num_heads=8, head_size=128)
comptime replit_num_q_heads = 24

comptime kv_params_llama3 = KVCacheStaticParams(num_heads=8, head_size=128)
comptime llama_num_q_heads = 32


def execute_ragged_flash_attention[
    num_q_heads: Int, dtype: DType, kv_params: KVCacheStaticParams
](
    valid_lengths: List[Int],
    max_seq_len_cache: Int,
    cache_lengths: List[Int],
    num_layers: Int,
    layer_idx: Int,
) raises:
    comptime num_continuous_blocks = 32
    comptime page_size = 512
    comptime num_paged_blocks = 512
    var batch_size = len(valid_lengths)
    debug_assert(
        batch_size < num_continuous_blocks,
        "batch_size passed to unit test (",
        batch_size,
        ") is larger than configured num_continuous_blocks (",
        num_continuous_blocks,
        ")",
    )
    assert len(valid_lengths) == len(
        cache_lengths
    ), "expected valid_lengths and cache_lengths size to be equal"

    var input_row_offsets_heap = List(length=batch_size + 1, fill=UInt32(0))
    var input_row_offsets = TileTensor(
        Span(input_row_offsets_heap), row_major(batch_size + 1)
    )
    var cache_lengths_nd_heap = List(length=batch_size, fill=UInt32(0))
    var cache_lengths_nd = TileTensor(
        Span(cache_lengths_nd_heap), row_major(batch_size)
    )

    var total_length = 0
    var max_full_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        input_row_offsets[i] = UInt32(total_length)
        cache_lengths_nd[i] = UInt32(cache_lengths[i])
        max_full_context_length = max(
            max_full_context_length, cache_lengths[i] + valid_lengths[i]
        )
        max_prompt_length = max(max_prompt_length, valid_lengths[i])
        total_length += valid_lengths[i]
    input_row_offsets[batch_size] = UInt32(total_length)

    var q_ragged_heap = List(
        length=total_length * num_q_heads * kv_params.head_size,
        fill=Scalar[dtype](0),
    )
    var q_ragged = TileTensor(
        Span(q_ragged_heap),
        row_major((total_length, Idx[num_q_heads], Idx[kv_params.head_size])),
    )
    random(q_ragged)

    # initialize reference output
    var test_output_heap = List(
        length=total_length * num_q_heads * kv_params.head_size,
        fill=Scalar[dtype](0),
    )
    var test_output = TileTensor(
        Span(test_output_heap),
        row_major((total_length, Idx[num_q_heads], Idx[kv_params.head_size])),
    )
    var ref_output_heap = List(
        length=total_length * num_q_heads * kv_params.head_size,
        fill=Scalar[dtype](0),
    )
    var ref_output = TileTensor(
        Span(ref_output_heap),
        row_major((total_length, Idx[num_q_heads], Idx[kv_params.head_size])),
    )

    # initialize our KVCache
    var block_shape = IndexList[6](
        num_continuous_blocks,
        2,
        num_layers,
        max_seq_len_cache,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var block_heap = List(
        length=block_shape.flattened_length(), fill=Scalar[dtype](0)
    )
    var kv_block_continuous = TileTensor(
        Span(block_heap), row_major(Coord(block_shape))
    )

    random(kv_block_continuous)

    var lookup_table_continuous_heap = List(length=batch_size, fill=UInt32(0))
    var lookup_table_continuous = TileTensor(
        Span(lookup_table_continuous_heap), row_major(batch_size)
    )

    # hacky way to select random blocks for continuous batching
    var block_idx_set = Set[Int]()
    var idx = 0
    while idx < batch_size:
        var randval = Int(random_ui64(0, num_continuous_blocks - 1))
        if randval in block_idx_set:
            continue

        block_idx_set.add(randval)
        lookup_table_continuous[idx] = UInt32(randval)
        idx += 1

    var kv_collection_continuous = ContinuousBatchingKVCacheCollection[
        dtype, kv_params
    ](
        kv_block_continuous,
        cache_lengths_nd.as_imm(),
        lookup_table_continuous.as_imm(),
        UInt32(max_prompt_length),
        UInt32(max_full_context_length),
    )

    var kv_block_paged_heap = List(
        length=num_paged_blocks
        * 2
        * num_layers
        * page_size
        * kv_params.num_heads
        * kv_params.head_size,
        fill=inf[dtype](),
    )
    var kv_block_paged = TileTensor(
        Span(kv_block_paged_heap),
        row_major(
            (
                Idx[num_paged_blocks],
                Idx[2],
                num_layers,
                Idx[page_size],
                Idx[kv_params.num_heads],
                Idx[kv_params.head_size],
            )
        ),
    )

    var paged_lut_heap = List(
        length=batch_size * ceildiv(max_full_context_length, page_size),
        fill=UInt32(0),
    )
    var paged_lut = TileTensor(
        Span(paged_lut_heap),
        row_major((batch_size, ceildiv(max_full_context_length, page_size))),
    )
    var paged_lut_set = Set[Int]()
    for bs in range(batch_size):
        var seq_len = cache_lengths[bs] + valid_lengths[bs]
        var continuous_idx = Int(lookup_table_continuous[bs])

        for block_idx in range(0, ceildiv(seq_len, page_size)):
            var randval = Int(random_ui64(0, num_paged_blocks - 1))
            while randval in paged_lut_set:
                randval = Int(random_ui64(0, num_paged_blocks - 1))

            paged_lut_set.add(randval)
            paged_lut[bs, block_idx] = UInt32(randval)

            for kv_idx in range(2):
                var dest = kv_block_paged.ptr_at_offset(
                    Coord(randval, kv_idx, layer_idx, 0, 0, 0)
                )
                var src = kv_block_continuous.ptr_at_offset(
                    Coord(
                        continuous_idx,
                        kv_idx,
                        layer_idx,
                        block_idx * page_size,
                        0,
                        0,
                    )
                )
                var dest_byte_offset = Int(dest) - Int(
                    kv_block_paged.unsafe_ptr()
                )
                var src_byte_offset = Int(src) - Int(
                    kv_block_continuous.unsafe_ptr()
                )
                var dest_remaining_bytes = (
                    kv_block_paged.num_elements() * size_of[dtype]()
                    - dest_byte_offset
                )
                var src_remaining_bytes = (
                    kv_block_continuous.num_elements() * size_of[dtype]()
                    - src_byte_offset
                )
                assert_true(dest_remaining_bytes >= 0)
                assert_true(src_remaining_bytes >= 0)
                # A partial source page must not read into the next KV block.
                var source_rows = min(
                    page_size, max_seq_len_cache - block_idx * page_size
                )
                var source_elements = (
                    source_rows * kv_params.num_heads * kv_params.head_size
                )
                var copy_count = min(
                    dest_remaining_bytes // size_of[dtype](),
                    src_remaining_bytes // size_of[dtype](),
                    source_elements,
                )
                assert_equal(copy_count, source_elements)
                unsafe_memcpy(dest=dest, src=src, count=copy_count)
                for i in range(copy_count):
                    assert_equal(dest[i], src[i])
                for i in range(
                    copy_count,
                    page_size * kv_params.num_heads * kv_params.head_size,
                ):
                    assert_equal(dest[i], inf[dtype]())

    var kv_collection_paged = PagedKVCacheCollection[
        dtype,
        kv_params,
        page_size,
        scales_origin=MutUntrackedOrigin,
    ](
        kv_block_paged,
        cache_lengths_nd.as_imm(),
        paged_lut.as_imm(),
        UInt32(max_prompt_length),
        UInt32(max_full_context_length),
    )

    # continuous execution
    flash_attention_kv_cache(
        q_ragged.as_imm(),
        input_row_offsets.as_imm(),
        # Assume self attention: Q and KV sequence lengths are equal.
        input_row_offsets.as_imm(),
        kv_collection_continuous.get_key_cache(layer_idx),
        kv_collection_continuous.get_value_cache(layer_idx),
        CausalMask(),
        rsqrt(Float32(kv_params.head_size)),
        ref_output,
    )

    # paged execution
    flash_attention_kv_cache(
        q_ragged.as_imm(),
        input_row_offsets.as_imm(),
        # Assume self attention: Q and KV sequence lengths are equal.
        input_row_offsets.as_imm(),
        kv_collection_paged.get_key_cache(layer_idx),
        kv_collection_paged.get_value_cache(layer_idx),
        CausalMask(),
        rsqrt(Float32(kv_params.head_size)),
        test_output,
    )

    var ref_out = ref_output
    var test_out = test_output
    for bs in range(batch_size):
        var prompt_len = valid_lengths[bs]
        var ragged_offset = Int(input_row_offsets[bs])
        for s in range(prompt_len):
            for h in range(num_q_heads):
                for hd in range(kv_params.head_size):
                    try:
                        assert_almost_equal(
                            ref_out[ragged_offset + s, h, hd][0],
                            test_out[ragged_offset + s, h, hd][0],
                            atol=1e-2,
                        )
                    except e:
                        print(
                            "MISMATCH:",
                            bs,
                            s,
                            h,
                            hd,
                            ref_out[ragged_offset + s, h, hd][0],
                            test_out[ragged_offset + s, h, hd][0],
                        )
                        raise e^
    _ = lookup_table_continuous_heap^
    _ = block_heap^
    _ = ref_output_heap^
    _ = test_output_heap^
    _ = q_ragged_heap^
    _ = cache_lengths_nd_heap^
    _ = input_row_offsets_heap^


comptime dtype = DType.float32


def execute_flash_attention_suite() raises:
    for bs in [1, 16]:
        var ce_cache_sizes = List[Int]()
        var ce_seq_lens = List[Int]()
        var tg_cache_sizes = List[Int]()
        var tg_seq_lens = List[Int]()
        for _ in range(bs):
            tg_seq_lens.append(1)
            tg_cache_sizes.append(Int(random_ui64(1, 100)))
            ce_seq_lens.append(Int(random_ui64(2, 100)))
            ce_cache_sizes.append(0)

        print("CE", bs, dtype)
        execute_ragged_flash_attention[
            llama_num_q_heads, dtype, kv_params_llama3
        ](ce_seq_lens, 110, ce_cache_sizes, 2, 1)

        print("TG", bs, dtype)
        execute_ragged_flash_attention[
            llama_num_q_heads, dtype, kv_params_llama3
        ](tg_seq_lens, 110, tg_cache_sizes, 2, 0)

    # edge cases
    var short_ce_seq_len: List = [2]
    var short_ce_cache_size: List = [0]
    execute_ragged_flash_attention[llama_num_q_heads, dtype, kv_params_llama3](
        short_ce_seq_len, 110, short_ce_cache_size, 2, 1
    )
    # Cross a page boundary with one complete and one partial source page.
    execute_ragged_flash_attention[llama_num_q_heads, dtype, kv_params_llama3](
        [2], 515, [513], 2, 1
    )


def main() raises:
    seed(42)
    execute_flash_attention_suite()
