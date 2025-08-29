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
from math import isqrt
from random import random_ui64, seed

from buffer import Dim, DimList
from internal_utils import HostNDBuffer, random
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
)
from memory import memcpy
from nn.flash_attention import flash_attention_kv_cache
from nn.mha_mask import CausalMask
from testing import assert_almost_equal

from utils import IndexList

alias kv_params_llama3 = KVCacheStaticParams(num_heads=8, head_size=128)
alias llama_num_q_heads = 32


def execute_ragged_flash_attention[
    num_q_heads: Int, dtype: DType, kv_params: KVCacheStaticParams
](
    valid_lengths_list: List[Int],
    max_seq_len_cache: Int,
    cache_lengths_list: List[Int],
    num_layers: Int,
    layer_idx: Int,
):
    alias num_blocks = 32
    alias CollectionType = ContinuousBatchingKVCacheCollection[dtype, kv_params]

    var batch_size = len(valid_lengths_list)
    debug_assert(
        batch_size < num_blocks,
        "batch_size passed to unit test (",
        batch_size,
        ") is larger than configured num_blocks (",
        num_blocks,
        ")",
    )
    debug_assert(
        len(valid_lengths_list) == len(cache_lengths_list),
        "expected valid_lengths and cache_lengths size to be equal",
    )

    var input_row_offsets = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size + 1)
    )
    var cache_lengths = HostNDBuffer[DType.uint32, 1](IndexList[1](batch_size))
    var valid_lengths = HostNDBuffer[DType.uint32, 1](IndexList[1](batch_size))

    var total_length = 0
    var max_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        input_row_offsets.tensor[i] = total_length
        cache_lengths.tensor[i] = cache_lengths_list[i]
        valid_lengths.tensor[i] = valid_lengths_list[i]
        max_context_length = max(
            max_context_length, cache_lengths_list[i] + valid_lengths_list[i]
        )
        max_prompt_length = max(max_prompt_length, valid_lengths_list[i])
        total_length += valid_lengths_list[i]
    input_row_offsets.tensor[batch_size] = total_length

    q_ragged = HostNDBuffer[
        dtype, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](total_length, num_q_heads, kv_params.head_size))
    random(q_ragged.tensor)
    q_padded = HostNDBuffer[
        dtype, 4, DimList(Dim(), Dim(), num_q_heads, kv_params.head_size)
    ](
        IndexList[4](
            batch_size, max_prompt_length, num_q_heads, kv_params.head_size
        )
    )

    # copy over the ragged values to the padded tensor.
    # Don't worry about padded values, we won't read them.
    for bs in range(batch_size):
        unpadded_seq_len = valid_lengths_list[bs]
        ragged_start_idx = Int(input_row_offsets.tensor[bs])
        padded_ptr = q_padded.tensor._offset(IndexList[4](bs, 0, 0, 0))
        ragged_ptr = q_ragged.tensor._offset(
            IndexList[3](ragged_start_idx, 0, 0)
        )
        memcpy(
            padded_ptr,
            ragged_ptr,
            unpadded_seq_len * num_q_heads * kv_params.head_size,
        )

    # initialize reference output
    ref_output = HostNDBuffer[
        dtype, 4, DimList(Dim(), Dim(), num_q_heads, kv_params.head_size)
    ](
        IndexList[4](
            batch_size, max_prompt_length, num_q_heads, kv_params.head_size
        ),
    )

    test_output = HostNDBuffer[
        dtype, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](total_length, num_q_heads, kv_params.head_size))

    # initialize our KVCache
    kv_block = HostNDBuffer[dtype, 6](
        IndexList[6](
            num_blocks,
            2,
            num_layers,
            max_seq_len_cache,
            kv_params.num_heads,
            kv_params.head_size,
        ),
    )
    random(kv_block.tensor)
    var lookup_table = HostNDBuffer[DType.uint32, 1](
        IndexList[1](
            batch_size,
        ),
    )

    # hacky way to select random blocks.
    var block_idx_set = Set[Int]()
    var idx = 0
    while idx < batch_size:
        var randval = Int(random_ui64(0, num_blocks - 1))
        if randval in block_idx_set:
            continue

        block_idx_set.add(randval)
        lookup_table.tensor[idx] = UInt32(randval)
        idx += 1

    var kv_collection = CollectionType(
        kv_block.tensor,
        cache_lengths.tensor,
        lookup_table.tensor,
        max_prompt_length,
        max_context_length,
    )

    var k_cache = kv_collection.get_key_cache(layer_idx)
    var v_cache = kv_collection.get_value_cache(layer_idx)

    # ragged execution
    flash_attention_kv_cache(
        q_ragged.tensor,
        input_row_offsets.tensor,
        # Assume self attention: Q and KV sequence lengths are equal.
        input_row_offsets.tensor,
        k_cache,
        v_cache,
        CausalMask(),
        isqrt(Float32(kv_params.head_size)),
        test_output.tensor,
    )
    # padded execution
    flash_attention_kv_cache(
        q_padded.tensor,
        k_cache,
        v_cache,
        CausalMask(),
        isqrt(Float32(kv_params.head_size)),
        ref_output.tensor,
    )

    ref_out = ref_output.tensor
    test_out = test_output.tensor
    for bs in range(batch_size):
        prompt_len = Int(valid_lengths.tensor[bs])
        ragged_offset = Int(input_row_offsets.tensor[bs])
        for s in range(prompt_len):
            for h in range(num_q_heads):
                for hd in range(kv_params.head_size):
                    try:
                        assert_almost_equal(
                            ref_out[bs, s, h, hd],
                            test_out[ragged_offset + s, h, hd],
                        )
                    except e:
                        print(
                            "MISMATCH:",
                            bs,
                            s,
                            h,
                            hd,
                            ref_out[bs, s, h, hd],
                            test_out[ragged_offset + s, h, hd],
                        )
                        raise e

    _ = q_ragged^
    _ = q_padded^
    _ = kv_block^
    _ = lookup_table^
    _ = ref_output^
    _ = test_output^
    _ = valid_lengths^
    _ = cache_lengths^


alias dtype = DType.float32


def execute_flash_attention_suite():
    for bs in [1, 16]:
        ce_cache_sizes = List[Int]()
        ce_seq_lens = List[Int]()
        tg_cache_sizes = List[Int]()
        tg_seq_lens = List[Int]()
        for _ in range(bs):
            tg_seq_lens.append(1)
            tg_cache_sizes.append(Int(random_ui64(1, 100)))
            ce_seq_lens.append(Int(random_ui64(1, 100)))
            ce_cache_sizes.append(0)
        print("CE", bs, dtype)
        execute_ragged_flash_attention[
            llama_num_q_heads, dtype, kv_params_llama3
        ](ce_seq_lens, 110, ce_cache_sizes, 2, 1)

        print("TG", bs, dtype)
        execute_ragged_flash_attention[
            llama_num_q_heads, dtype, kv_params_llama3
        ](tg_seq_lens, 110, tg_cache_sizes, 2, 0)

    # Edge-case specific tests
    # Case 0: token gen in one batch, context encoding in another
    var c0_seq_lens = List[Int](25, 1)
    var c0_cache_sizes = List[Int](0, 25)

    execute_ragged_flash_attention[llama_num_q_heads, dtype, kv_params_llama3](
        c0_seq_lens, 110, c0_cache_sizes, 2, 0
    )


def main():
    seed(42)
    execute_flash_attention_suite()
