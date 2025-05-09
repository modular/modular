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

from collections import Set
from math import ceildiv, isclose, isqrt
from random import random_ui64, seed

from buffer import Dim, DimList, NDBuffer
from internal_utils import HostNDBuffer, fill, random
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from memory import UnsafePointer, memcpy
from nn.flash_attention import flash_attention_kv_cache
from nn.mha_mask import CausalMask, NullMask
from nn.mha_score_mod import IdentityScoreMod
from testing import assert_almost_equal

from utils import Index, IndexList

alias kv_params_replit = KVCacheStaticParams(num_heads=8, head_size=128)
alias replit_num_q_heads = 24

alias kv_params_llama3 = KVCacheStaticParams(num_heads=8, head_size=128)
alias llama_num_q_heads = 32


def execute_ragged_flash_attention[
    num_q_heads: Int, type: DType, kv_params: KVCacheStaticParams
](
    valid_lengths: List[Int],
    max_seq_len_cache: Int,
    cache_lengths: List[Int],
    num_layers: Int,
    layer_idx: Int,
):
    alias num_continuous_blocks = 32
    alias page_size = 512
    alias num_paged_blocks = 512
    var batch_size = len(valid_lengths)
    debug_assert(
        batch_size < num_continuous_blocks,
        "batch_size passed to unit test (",
        batch_size,
        ") is larger than configured num_continuous_blocks (",
        num_continuous_blocks,
        ")",
    )
    debug_assert(
        len(valid_lengths) == len(cache_lengths),
        "expected valid_lengths and cache_lengths size to be equal",
    )

    var input_row_offsets = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size + 1)
    )
    var cache_lengths_nd = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size)
    )

    var total_length = 0
    var max_full_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        input_row_offsets.tensor[i] = total_length
        cache_lengths_nd.tensor[i] = cache_lengths[i]
        max_full_context_length = max(
            max_full_context_length, cache_lengths[i] + valid_lengths[i]
        )
        max_prompt_length = max(max_prompt_length, valid_lengths[i])
        total_length += valid_lengths[i]
    input_row_offsets.tensor[batch_size] = total_length

    q_ragged = HostNDBuffer[
        type, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](total_length, num_q_heads, kv_params.head_size))
    random(q_ragged.tensor)

    # initialize reference output
    test_output = HostNDBuffer[
        type, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](total_length, num_q_heads, kv_params.head_size))
    ref_output = HostNDBuffer[
        type, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](total_length, num_q_heads, kv_params.head_size))

    # initialize our KVCache
    kv_block_continuous = HostNDBuffer[type, 6](
        IndexList[6](
            num_continuous_blocks,
            2,
            num_layers,
            max_seq_len_cache,
            kv_params.num_heads,
            kv_params.head_size,
        ),
    )

    random(kv_block_continuous.tensor)
    var lookup_table_continuous = HostNDBuffer[DType.uint32, 1](
        IndexList[1](
            batch_size,
        ),
    )

    # hacky way to select random blocks for continuous batching
    var block_idx_set = Set[Int]()
    var idx = 0
    while idx < batch_size:
        var randval = Int(random_ui64(0, num_continuous_blocks - 1))
        if randval in block_idx_set:
            continue

        block_idx_set.add(randval)
        lookup_table_continuous.tensor[idx] = UInt32(randval)
        idx += 1

    kv_collection_continuous = ContinuousBatchingKVCacheCollection[
        type, kv_params
    ](
        kv_block_continuous.tensor,
        cache_lengths_nd.tensor,
        lookup_table_continuous.tensor,
        max_prompt_length,
        max_full_context_length,
    )

    kv_block_paged = HostNDBuffer[type, 6](
        IndexList[6](
            num_paged_blocks,
            2,
            num_layers,
            page_size,
            kv_params.num_heads,
            kv_params.head_size,
        )
    )

    paged_lut = HostNDBuffer[DType.uint32, 2](
        IndexList[2](batch_size, ceildiv(max_full_context_length, page_size))
    )
    paged_lut_set = Set[Int]()
    for bs in range(batch_size):
        seq_len = cache_lengths[bs] + valid_lengths[bs]
        continuous_idx = Int(lookup_table_continuous.tensor[bs])

        for block_idx in range(0, ceildiv(seq_len, page_size)):
            var randval = Int(random_ui64(0, num_paged_blocks - 1))
            while randval in paged_lut_set:
                randval = Int(random_ui64(0, num_paged_blocks - 1))

            paged_lut_set.add(randval)
            paged_lut.tensor[bs, block_idx] = randval

            for kv_idx in range(2):
                memcpy(
                    kv_block_paged.tensor._offset(
                        IndexList[6](randval, kv_idx, layer_idx, 0, 0, 0)
                    ),
                    kv_block_continuous.tensor._offset(
                        IndexList[6](
                            continuous_idx,
                            kv_idx,
                            layer_idx,
                            block_idx * page_size,
                            0,
                            0,
                        )
                    ),
                    page_size * kv_params.num_heads * kv_params.head_size,
                )

    kv_collection_paged = PagedKVCacheCollection[type, kv_params, page_size](
        kv_block_paged.tensor,
        cache_lengths_nd.tensor,
        paged_lut.tensor,
        max_prompt_length,
        max_full_context_length,
    )

    # continuous execution
    flash_attention_kv_cache(
        q_ragged.tensor,
        input_row_offsets.tensor,
        # Assume self attention: Q and KV sequence lengths are equal.
        input_row_offsets.tensor,
        kv_collection_continuous.get_key_cache(layer_idx),
        kv_collection_continuous.get_value_cache(layer_idx),
        CausalMask(),
        isqrt(Float32(kv_params.head_size)),
        ref_output.tensor,
    )

    # paged execution
    flash_attention_kv_cache(
        q_ragged.tensor,
        input_row_offsets.tensor,
        # Assume self attention: Q and KV sequence lengths are equal.
        input_row_offsets.tensor,
        kv_collection_paged.get_key_cache(layer_idx),
        kv_collection_paged.get_value_cache(layer_idx),
        CausalMask(),
        isqrt(Float32(kv_params.head_size)),
        test_output.tensor,
    )

    ref_out = ref_output.tensor
    test_out = test_output.tensor
    for bs in range(batch_size):
        prompt_len = valid_lengths[bs]
        ragged_offset = Int(input_row_offsets.tensor[bs])
        for s in range(prompt_len):
            for h in range(num_q_heads):
                for hd in range(kv_params.head_size):
                    try:
                        assert_almost_equal(
                            ref_out[ragged_offset + s, h, hd],
                            test_out[ragged_offset + s, h, hd],
                            atol=1e-2,
                        )
                    except e:
                        print(
                            "MISMATCH:",
                            bs,
                            s,
                            h,
                            hd,
                            ref_out[ragged_offset + s, h, hd],
                            test_out[ragged_offset + s, h, hd],
                        )
                        raise e

    _ = q_ragged^
    _ = kv_block_continuous^
    _ = kv_block_paged^
    _ = lookup_table_continuous^
    _ = ref_output^
    _ = test_output^
    _ = cache_lengths_nd^
    _ = paged_lut^


alias type = DType.float32


def execute_flash_attention_suite():
    for bs_ref in List[Int](1, 16):
        bs = bs_ref[]
        ce_cache_sizes = List[Int]()
        ce_seq_lens = List[Int]()
        tg_cache_sizes = List[Int]()
        tg_seq_lens = List[Int]()
        for _ in range(bs):
            tg_seq_lens.append(1)
            tg_cache_sizes.append(Int(random_ui64(1, 100)))
            ce_seq_lens.append(Int(random_ui64(1, 100)))
            ce_cache_sizes.append(0)

        print("CE", bs, type)
        execute_ragged_flash_attention[
            llama_num_q_heads, type, kv_params_llama3
        ](ce_seq_lens, 110, ce_cache_sizes, 2, 1)

        print("TG", bs, type)
        execute_ragged_flash_attention[
            llama_num_q_heads, type, kv_params_llama3
        ](tg_seq_lens, 110, tg_cache_sizes, 2, 0)

    # edge cases
    var short_ce_seq_len = List[Int](2)
    var short_ce_cache_size = List[Int](0)
    execute_ragged_flash_attention[llama_num_q_heads, type, kv_params_llama3](
        short_ce_seq_len, 110, short_ce_cache_size, 2, 1
    )


def main():
    seed(42)
    execute_flash_attention_suite()
