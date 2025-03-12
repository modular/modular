# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug-no-assert %s

from collections import Set
from math import ceildiv, isclose, isqrt
from random import random_ui64, seed

from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer, fill, random
from kv_cache.types import (
    ContinuousBatchingKVCache,
    KVCacheStaticParams,
    PagedKVCache,
)
from memory import UnsafePointer, memcpy
from nn.mha import flash_attention
from nn.mha_mask import CausalMask, NullMask
from nn.mha_score_mod import IdentityScoreMod
from testing import assert_almost_equal

from utils import IndexList
from utils.index import Index

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
    ctx: DeviceContext,
):
    alias num_continuous_blocks = 32
    alias num_paged_blocks = 32
    alias page_size = 512
    alias PagedCacheType = PagedKVCache[type, kv_params, page_size]
    alias ContinuousBatchCacheType = ContinuousBatchingKVCache[type, kv_params]
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

    var input_row_offsets_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size + 1)
    )
    var cache_lengths_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size)
    )

    var total_length = 0
    var max_full_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        input_row_offsets_host.tensor[i] = total_length
        cache_lengths_host.tensor[i] = cache_lengths[i]
        max_full_context_length = max(
            max_full_context_length, cache_lengths[i] + valid_lengths[i]
        )
        max_prompt_length = max(max_prompt_length, valid_lengths[i])
        total_length += valid_lengths[i]
    input_row_offsets_host.tensor[batch_size] = total_length

    input_row_offsets_device = input_row_offsets_host.copy_to_device(ctx)
    cache_lengths_device = cache_lengths_host.copy_to_device(ctx)

    q_ragged_host = HostNDBuffer[
        type, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](total_length, num_q_heads, kv_params.head_size))
    random(q_ragged_host.tensor)
    q_ragged_device = q_ragged_host.copy_to_device(ctx)

    # initialize mask tensor
    # dummy mask to satisfy the argument.
    dummy_mask = NDBuffer[type, 4](
        UnsafePointer[Scalar[type]](), IndexList[4]()
    )

    # initialize scale tensor
    scale_host = HostNDBuffer[DType.float32, 1, DimList(1)](IndexList[1](1))

    scale_host.tensor[0] = isqrt(Float32(kv_params.head_size))
    scale_device = scale_host.copy_to_device(ctx)

    # initialize reference output
    test_output_host = HostNDBuffer[
        type, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](total_length, num_q_heads, kv_params.head_size))
    test_output_device = test_output_host.copy_to_device(ctx)
    ref_output_host = HostNDBuffer[
        type, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](total_length, num_q_heads, kv_params.head_size))
    ref_output_device = ref_output_host.copy_to_device(ctx)

    # initialize our KVCache
    kv_block_continuous_host = HostNDBuffer[type, 6](
        IndexList[6](
            num_continuous_blocks,
            2,
            num_layers,
            max_seq_len_cache,
            kv_params.num_heads,
            kv_params.head_size,
        ),
    )

    random(kv_block_continuous_host.tensor)
    kv_block_continuous_device = kv_block_continuous_host.copy_to_device(ctx)
    var lookup_table_continuous_host = HostNDBuffer[DType.uint32, 1](
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
        lookup_table_continuous_host.tensor[idx] = UInt32(randval)
        idx += 1
    var lookup_table_device = lookup_table_continuous_host.copy_to_device(ctx)
    var k_cache_continuous_device = ContinuousBatchCacheType(
        kv_block_continuous_device.tensor,
        cache_lengths_device.tensor,
        lookup_table_device.tensor,
        max_prompt_length,
        max_full_context_length,
        layer_idx,
        ContinuousBatchCacheType.KeyIdx,
    )
    var v_cache_continuous_device = ContinuousBatchCacheType(
        kv_block_continuous_device.tensor,
        cache_lengths_device.tensor,
        lookup_table_device.tensor,
        max_prompt_length,
        max_full_context_length,
        layer_idx,
        ContinuousBatchCacheType.ValueIdx,
    )

    kv_block_paged_host = HostNDBuffer[type, 6](
        IndexList[6](
            num_paged_blocks,
            2,
            num_layers,
            page_size,
            kv_params.num_heads,
            kv_params.head_size,
        )
    )

    paged_lut_host = HostNDBuffer[DType.uint32, 2](
        IndexList[2](batch_size, ceildiv(max_full_context_length, page_size))
    )
    paged_lut_set = Set[Int]()
    for bs in range(batch_size):
        seq_len = cache_lengths[bs] + valid_lengths[bs]
        continuous_idx = Int(lookup_table_continuous_host.tensor[bs])

        for block_idx in range(0, ceildiv(seq_len, page_size)):
            var randval = Int(random_ui64(0, num_paged_blocks - 1))
            while randval in paged_lut_set:
                randval = Int(random_ui64(0, num_paged_blocks - 1))

            paged_lut_set.add(randval)
            paged_lut_host.tensor[bs, block_idx] = randval

            for kv_idx in range(2):
                memcpy(
                    kv_block_paged_host.tensor._offset(
                        IndexList[6](randval, kv_idx, layer_idx, 0, 0, 0)
                    ),
                    kv_block_continuous_host.tensor._offset(
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
    paged_lut_device = paged_lut_host.copy_to_device(ctx)
    kv_block_paged_device = kv_block_paged_host.copy_to_device(ctx)

    k_cache_paged_device = PagedCacheType(
        kv_block_paged_device.tensor,
        cache_lengths_device.tensor,
        paged_lut_device.tensor,
        max_prompt_length,
        max_full_context_length,
        layer_idx,
        PagedCacheType.KeyIdx,
    )

    v_cache_paged_device = PagedCacheType(
        kv_block_paged_device.tensor,
        cache_lengths_device.tensor,
        paged_lut_device.tensor,
        max_prompt_length,
        max_full_context_length,
        layer_idx,
        PagedCacheType.ValueIdx,
    )

    # continuous execution
    flash_attention[add_attn_mask=False, ragged=True](
        ref_output_device.tensor,
        q_ragged_device.tensor,
        k_cache_continuous_device,
        v_cache_continuous_device,
        dummy_mask,
        CausalMask(),
        IdentityScoreMod(),
        input_row_offsets_device.tensor,
        # TODO take scale from argument GRA-750
        isqrt(Float32(kv_params.head_size)),
        ctx,
    )

    # paged execution
    flash_attention[add_attn_mask=False, ragged=True](
        test_output_device.tensor,
        q_ragged_device.tensor,
        k_cache_paged_device,
        v_cache_paged_device,
        dummy_mask,
        CausalMask(),
        IdentityScoreMod(),
        input_row_offsets_device.tensor,
        # TODO take scale from argument GRA-750
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

    _ = q_ragged_host^
    _ = q_ragged_device^
    _ = scale_host^
    _ = scale_device^
    _ = kv_block_continuous_host^
    _ = kv_block_continuous_device^
    _ = kv_block_paged_host^
    _ = kv_block_paged_device^
    _ = lookup_table_continuous_host^
    _ = lookup_table_device^
    _ = ref_output_device^
    _ = ref_output_host^
    _ = test_output_device^
    _ = test_output_host^
    _ = cache_lengths_host^
    _ = cache_lengths_device^
    _ = paged_lut_host^
    _ = paged_lut_device^


def execute_flash_attention_suite(ctx: DeviceContext):
    alias types = (DType.float32, DType.bfloat16)

    for bs_ref in List[Int](1, 16):

        @parameter
        for type_idx in range(2):
            alias type = types[type_idx]
            bs = bs_ref[]
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
            ](ce_seq_lens, 1500, ce_cache_sizes, 2, 1, ctx)

            print("TG", bs, type)
            execute_ragged_flash_attention[
                llama_num_q_heads, type, kv_params_llama3
            ](tg_seq_lens, 1500, tg_cache_sizes, 2, 0, ctx)

    # edge cases
    var short_ce_seq_len = List[Int](2)
    var short_ce_cache_size = List[Int](0)
    execute_ragged_flash_attention[
        llama_num_q_heads, DType.bfloat16, kv_params_llama3
    ](short_ce_seq_len, 110, short_ce_cache_size, 2, 1, ctx)


def main():
    seed(42)
    with DeviceContext() as ctx:
        execute_flash_attention_suite(ctx)
