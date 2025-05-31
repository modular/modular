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
from math import isclose, isqrt
from random import random_ui64, seed
from sys import has_nvidia_gpu_accelerator

from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer, random
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
)
from memory import UnsafePointer, memcpy
from nn.mha import flash_attention
from nn.mha_mask import CausalMask
from nn.mha_score_mod import IdentityScoreMod
from tensor_internal import IOUnknown, ManagedTensorSlice
from tensor_internal.managed_tensor_slice import StaticTensorSpec
from testing import assert_almost_equal

from utils import Index, IndexList

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
    ctx: DeviceContext,
):
    alias num_blocks = 32
    alias CollectionType = ContinuousBatchingKVCacheCollection[
        type, kv_params, WRITE_MODE_MEM
    ]

    var batch_size = len(valid_lengths)
    debug_assert[WRITE_MODE_MEM](
        batch_size < num_blocks,
        "batch_size passed to unit test (",
        batch_size,
        ") is larger than configured num_blocks (",
        num_blocks,
        ")",
    )
    debug_assert[WRITE_MODE_MEM](
        len(valid_lengths) == len(cache_lengths),
        "expected valid_lengths and cache_lengths size to be equal",
    )

    var input_row_offsets_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size + 1)
    )
    var cache_lengths_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size)
    )
    var valid_lengths_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size)
    )

    var total_length = 0
    var max_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        input_row_offsets_host.tensor[i] = total_length
        cache_lengths_host.tensor[i] = cache_lengths[i]
        valid_lengths_host.tensor[i] = valid_lengths[i]
        max_context_length = max(
            max_context_length, cache_lengths[i] + valid_lengths[i]
        )
        max_prompt_length = max(max_prompt_length, valid_lengths[i])
        total_length += valid_lengths[i]
    input_row_offsets_host.tensor[batch_size] = total_length

    input_row_offsets_device = input_row_offsets_host.copy_to_device(ctx)
    valid_lengths_device = valid_lengths_host.copy_to_device(ctx)
    cache_lengths_device = cache_lengths_host.copy_to_device(ctx)

    q_ragged_host = HostNDBuffer[
        type, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](total_length, num_q_heads, kv_params.head_size))
    random(q_ragged_host.tensor)
    q_padded_host = HostNDBuffer[
        type, 4, DimList(Dim(), Dim(), num_q_heads, kv_params.head_size)
    ](
        IndexList[4](
            batch_size, max_prompt_length, num_q_heads, kv_params.head_size
        )
    )

    # copy over the ragged values to the padded tensor.
    # Don't worry about padded values, we won't read them.
    for bs in range(batch_size):
        unpadded_seq_len = valid_lengths[bs]
        ragged_start_idx = Int(input_row_offsets_host.tensor[bs])
        padded_ptr = q_padded_host.tensor._offset((bs, 0, 0, 0))
        ragged_ptr = q_ragged_host.tensor._offset((ragged_start_idx, 0, 0))
        memcpy(
            padded_ptr,
            ragged_ptr,
            unpadded_seq_len * num_q_heads * kv_params.head_size,
        )

    q_ragged_device = q_ragged_host.copy_to_device(ctx)
    q_padded_device = q_padded_host.copy_to_device(ctx)

    # initialize reference output
    ref_output_host = HostNDBuffer[
        type, 4, DimList(Dim(), Dim(), num_q_heads, kv_params.head_size)
    ](
        IndexList[4](
            batch_size, max_prompt_length, num_q_heads, kv_params.head_size
        ),
    )
    ref_output_device = ref_output_host.copy_to_device(ctx)

    test_output_host = HostNDBuffer[
        type, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](total_length, num_q_heads, kv_params.head_size))
    test_output_device = test_output_host.copy_to_device(ctx)

    # initialize our KVCache
    kv_block_host = HostNDBuffer[type, 6](
        IndexList[6](
            num_blocks,
            2,
            num_layers,
            max_seq_len_cache,
            kv_params.num_heads,
            kv_params.head_size,
        ),
    )
    random(kv_block_host.tensor)
    kv_block_device = kv_block_host.copy_to_device(ctx)
    var lookup_table_host = HostNDBuffer[DType.uint32, 1](
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
        lookup_table_host.tensor[idx] = UInt32(randval)
        idx += 1

    var lookup_table_device = lookup_table_host.copy_to_device(ctx)

    var kv_collection_device = CollectionType(
        kv_block_device.tensor,
        cache_lengths_device.tensor,
        lookup_table_device.tensor,
        max_prompt_length,
        max_context_length,
    )
    var k_cache_device = kv_collection_device.get_key_cache(layer_idx)
    var v_cache_device = kv_collection_device.get_value_cache(layer_idx)

    # ragged execution
    flash_attention[ragged=True](
        test_output_device.tensor,
        q_ragged_device.tensor,
        k_cache_device,
        v_cache_device,
        CausalMask(),
        IdentityScoreMod(),
        ManagedTensorSlice[
            io_spec=IOUnknown,
            static_spec = StaticTensorSpec[DType.uint32, 1].create_unknown(),
        ](input_row_offsets_device.tensor),
        isqrt(Float32(kv_params.head_size)),
        ctx,
    )
    # padded execution
    flash_attention(
        ref_output_device.tensor,
        q_padded_device.tensor,
        k_cache_device,
        v_cache_device,
        CausalMask(),
        IdentityScoreMod(),
        ManagedTensorSlice[
            io_spec=IOUnknown,
            static_spec = StaticTensorSpec[DType.uint32, 1].create_unknown(),
        ](valid_lengths_device.tensor),
        isqrt(Float32(kv_params.head_size)),
        ctx,
    )

    ctx.enqueue_copy(test_output_host.tensor.data, test_output_device.buffer)
    ctx.enqueue_copy(ref_output_host.tensor.data, ref_output_device.buffer)
    ctx.synchronize()

    ref_out = ref_output_host.tensor
    test_out = test_output_host.tensor
    for bs in range(batch_size):
        prompt_len = valid_lengths[bs]
        ragged_offset = Int(input_row_offsets_host.tensor[bs])
        for s in range(prompt_len):
            for h in range(num_q_heads):
                for hd in range(kv_params.head_size):
                    var ref_val = ref_out[bs, s, h, hd]
                    var test_val = test_out[ragged_offset + s, h, hd]
                    try:
                        assert_almost_equal(
                            ref_val,
                            test_val,
                            rtol=1e-2 if type is DType.bfloat16 else 1e-4,
                        )
                    except e:
                        print(
                            "MISMATCH:",
                            bs,
                            s,
                            h,
                            hd,
                            ref_val,
                            test_val,
                        )
                        raise e

    _ = q_ragged_host^
    _ = q_ragged_device^
    _ = q_padded_host^
    _ = q_padded_device^
    _ = kv_block_host^
    _ = kv_block_device^
    _ = lookup_table_host^
    _ = lookup_table_device^
    _ = ref_output_device^
    _ = ref_output_host^
    _ = test_output_device^
    _ = test_output_host^
    _ = valid_lengths_device^
    _ = valid_lengths_host^
    _ = cache_lengths_host^
    _ = cache_lengths_device^
    _ = input_row_offsets_host^
    _ = input_row_offsets_device^


def execute_flash_attention_suite(ctx: DeviceContext):
    alias types = (DType.float32, DType.bfloat16)

    for bs in [1, 16]:

        @parameter
        for type_idx in range(len(types)):
            alias type = types[type_idx]

            ce_cache_sizes = List[Int]()
            ce_seq_lens = List[Int]()
            tg_cache_sizes = List[Int]()
            tg_seq_lens = List[Int]()
            for _ in range(bs):
                tg_seq_lens.append(1)
                tg_cache_sizes.append(Int(random_ui64(512, 1024)))
                ce_seq_lens.append(Int(random_ui64(512, 1024)))
                ce_cache_sizes.append(0)
            print("CE", bs, type)
            execute_ragged_flash_attention[
                llama_num_q_heads, type, kv_params_llama3
            ](ce_seq_lens, 1024, ce_cache_sizes, 2, 1, ctx)

            print("TG", bs, type)
            execute_ragged_flash_attention[
                llama_num_q_heads, type, kv_params_llama3
            ](tg_seq_lens, 1024, tg_cache_sizes, 2, 0, ctx)

    # edge cases
    var short_ce_seq_len = List[Int](2)
    var short_ce_cache_size = List[Int](0)
    execute_ragged_flash_attention[
        llama_num_q_heads, DType.bfloat16, kv_params_llama3
    ](short_ce_seq_len, 1024, short_ce_cache_size, 2, 1, ctx)


def main():
    seed(42)
    with DeviceContext() as ctx:
        execute_flash_attention_suite(ctx)
