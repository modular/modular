# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s -t

from collections import Set
from math import ceildiv, isqrt
from random import random_ui64

from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer, random
from kv_cache.types import KVCacheStaticParams, PagedKVCache
from memory import UnsafePointer, memcpy
from nn.mha import flash_attention
from nn.mha_mask import CausalMask, NullMask
from nn.mha_score_mod import IdentityScoreMod
from testing import assert_almost_equal

from utils import IndexList


def execute_ragged_flash_attention(
    ctx: DeviceContext,
):
    alias num_q_heads = 32
    alias kv_params = KVCacheStaticParams(num_heads=8, head_size=128)
    alias type = DType.float32
    alias num_paged_blocks = 32
    alias page_size = 128
    alias PagedCacheType = PagedKVCache[type, kv_params, page_size]
    var num_layers = 1
    var layer_idx = 0

    var true_ce_prompt_lens = List[Int](100, 200, 300, 400)
    var mixed_ce_prompt_lens = List[Int](50, 100, 150, 100)

    var true_ce_cache_lens = List[Int](0, 0, 0, 0)
    var mixed_ce_cache_lens = List[Int](50, 100, 150, 300)

    var batch_size = len(true_ce_prompt_lens)

    var true_ce_row_offsets_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size + 1)
    )
    var true_ce_cache_lengths_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size)
    )
    var mixed_ce_row_offsets_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size + 1)
    )
    var mixed_ce_cache_lengths_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size)
    )

    var true_ce_total_length = 0
    var mixed_ce_total_length = 0
    var true_ce_max_full_context_length = 0
    var mixed_ce_max_full_context_length = 0
    var true_ce_max_prompt_length = 0
    var mixed_ce_max_prompt_length = 0
    for i in range(batch_size):
        true_ce_row_offsets_host.tensor[i] = true_ce_total_length
        mixed_ce_row_offsets_host.tensor[i] = mixed_ce_total_length
        true_ce_cache_lengths_host.tensor[i] = true_ce_cache_lens[i]
        mixed_ce_cache_lengths_host.tensor[i] = mixed_ce_cache_lens[i]

        true_ce_max_full_context_length = max(
            true_ce_max_full_context_length,
            true_ce_cache_lens[i] + true_ce_prompt_lens[i],
        )
        mixed_ce_max_full_context_length = max(
            mixed_ce_max_full_context_length,
            mixed_ce_cache_lens[i] + mixed_ce_prompt_lens[i],
        )

        true_ce_max_prompt_length = max(
            true_ce_max_prompt_length, true_ce_prompt_lens[i]
        )
        mixed_ce_max_prompt_length = max(
            mixed_ce_max_prompt_length, mixed_ce_prompt_lens[i]
        )

        true_ce_total_length += true_ce_prompt_lens[i]
        mixed_ce_total_length += mixed_ce_prompt_lens[i]

    true_ce_row_offsets_host.tensor[batch_size] = true_ce_total_length
    mixed_ce_row_offsets_host.tensor[batch_size] = mixed_ce_total_length
    true_ce_row_offsets_device = true_ce_row_offsets_host.copy_to_device(ctx)
    mixed_ce_row_offsets_device = mixed_ce_row_offsets_host.copy_to_device(ctx)
    true_ce_cache_lengths_device = true_ce_cache_lengths_host.copy_to_device(
        ctx
    )
    mixed_ce_cache_lengths_device = mixed_ce_cache_lengths_host.copy_to_device(
        ctx
    )
    true_ce_q_ragged_host = HostNDBuffer[
        type, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](true_ce_total_length, num_q_heads, kv_params.head_size))
    random(true_ce_q_ragged_host.tensor)
    true_ce_q_ragged_device = true_ce_q_ragged_host.copy_to_device(ctx)

    mixed_ce_q_ragged_host = HostNDBuffer[
        type, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](mixed_ce_total_length, num_q_heads, kv_params.head_size))
    for bs_idx in range(batch_size):
        true_ce_prompt_len = true_ce_prompt_lens[bs_idx]
        mixed_ce_prompt_len = mixed_ce_prompt_lens[bs_idx]

        true_ce_row_offset = true_ce_row_offsets_host.tensor[bs_idx]
        mixed_ce_row_offset = mixed_ce_row_offsets_host.tensor[bs_idx]

        mixed_ce_cache_len = mixed_ce_cache_lens[bs_idx]

        true_ce_offset = true_ce_q_ragged_host.tensor._offset(
            IndexList[3](Int(true_ce_row_offset + mixed_ce_cache_len), 0, 0)
        )
        mixed_ce_offset = mixed_ce_q_ragged_host.tensor._offset(
            IndexList[3](Int(mixed_ce_row_offset), 0, 0)
        )

        memcpy(
            mixed_ce_offset,
            true_ce_offset,
            mixed_ce_prompt_len * num_q_heads * kv_params.head_size,
        )

    mixed_ce_q_ragged_device = mixed_ce_q_ragged_host.copy_to_device(ctx)

    # dummy mask to satisfy the argument.
    dummy_mask = NDBuffer[type, 4](
        UnsafePointer[Scalar[type]](), IndexList[4]()
    )

    # initialize scale tensor
    scale_host = HostNDBuffer[DType.float32, 1, DimList(1)](IndexList[1](1))

    scale_host.tensor[0] = isqrt(Float32(kv_params.head_size))
    scale_device = scale_host.copy_to_device(ctx)

    # initialize reference output
    mixed_ce_output_host = HostNDBuffer[
        type, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](mixed_ce_total_length, num_q_heads, kv_params.head_size))
    mixed_ce_output_device = mixed_ce_output_host.copy_to_device(ctx)
    true_ce_output_host = HostNDBuffer[
        type, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](true_ce_total_length, num_q_heads, kv_params.head_size))
    true_ce_output_device = true_ce_output_host.copy_to_device(ctx)

    # initialize our KVCache
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
    random(kv_block_paged_host.tensor)

    paged_lut_host = HostNDBuffer[DType.uint32, 2](
        IndexList[2](
            batch_size,
            ceildiv(true_ce_max_full_context_length, page_size),
        )
    )
    paged_lut_set = Set[Int]()
    for bs in range(batch_size):
        seq_len = true_ce_cache_lens[bs] + true_ce_prompt_lens[bs]

        for block_idx in range(0, ceildiv(seq_len, page_size)):
            var randval = Int(random_ui64(0, num_paged_blocks - 1))
            while randval in paged_lut_set:
                randval = Int(random_ui64(0, num_paged_blocks - 1))

            paged_lut_set.add(randval)
            paged_lut_host.tensor[bs, block_idx] = randval

    paged_lut_device = paged_lut_host.copy_to_device(ctx)
    kv_block_paged_device = kv_block_paged_host.copy_to_device(ctx)

    true_ce_k_cache_device = PagedCacheType(
        kv_block_paged_device.tensor,
        true_ce_cache_lengths_device.tensor,
        paged_lut_device.tensor,
        true_ce_max_prompt_length,
        true_ce_max_full_context_length,
        layer_idx,
        PagedCacheType.KeyIdx,
    )

    true_ce_v_cache_paged_device = PagedCacheType(
        kv_block_paged_device.tensor,
        true_ce_cache_lengths_device.tensor,
        paged_lut_device.tensor,
        true_ce_max_prompt_length,
        true_ce_max_full_context_length,
        layer_idx,
        PagedCacheType.ValueIdx,
    )

    mixed_ce_k_cache_device = PagedCacheType(
        kv_block_paged_device.tensor,
        mixed_ce_cache_lengths_device.tensor,
        paged_lut_device.tensor,
        mixed_ce_max_prompt_length,
        mixed_ce_max_full_context_length,
        layer_idx,
        PagedCacheType.KeyIdx,
    )

    mixed_ce_v_cache_paged_device = PagedCacheType(
        kv_block_paged_device.tensor,
        mixed_ce_cache_lengths_device.tensor,
        paged_lut_device.tensor,
        mixed_ce_max_prompt_length,
        mixed_ce_max_full_context_length,
        layer_idx,
        PagedCacheType.ValueIdx,
    )

    # "true CE" execution
    print("true")
    flash_attention[ragged=True](
        true_ce_output_device.tensor,
        true_ce_q_ragged_device.tensor,
        true_ce_k_cache_device,
        true_ce_v_cache_paged_device,
        CausalMask(),
        IdentityScoreMod(),
        true_ce_row_offsets_device.tensor,
        # TODO take scale from argument GRA-750
        isqrt(Float32(kv_params.head_size)),
        ctx,
    )
    ctx.synchronize()

    # "mixed CE" execution
    print("mixed")
    flash_attention[ragged=True](
        mixed_ce_output_device.tensor,
        mixed_ce_q_ragged_device.tensor,
        mixed_ce_k_cache_device,
        mixed_ce_v_cache_paged_device,
        CausalMask(),
        IdentityScoreMod(),
        mixed_ce_row_offsets_device.tensor,
        # TODO take scale from argument GRA-750
        isqrt(Float32(kv_params.head_size)),
        ctx,
    )
    ctx.synchronize()
    ctx.enqueue_copy(
        mixed_ce_output_host.tensor.data, mixed_ce_output_device.buffer
    )
    ctx.enqueue_copy(
        true_ce_output_host.tensor.data, true_ce_output_device.buffer
    )
    ctx.synchronize()

    true_ce_out = true_ce_output_host.tensor
    mixed_ce_out = mixed_ce_output_host.tensor
    for bs in range(batch_size):
        mixed_ce_prompt_len = mixed_ce_prompt_lens[bs]
        mixed_ce_row_offset = mixed_ce_row_offsets_host.tensor[bs]
        true_ce_row_offset = true_ce_row_offsets_host.tensor[bs]
        mixed_ce_cache_len = mixed_ce_cache_lens[bs]

        true_ce_ragged_offset = Int(true_ce_row_offset + mixed_ce_cache_len)
        mixed_ce_ragged_offset = Int(mixed_ce_row_offset)
        for s in range(mixed_ce_prompt_len):
            for h in range(num_q_heads):
                for hd in range(kv_params.head_size):
                    true_ce_val = true_ce_out[true_ce_ragged_offset + s, h, hd]
                    mixed_ce_val = mixed_ce_out[
                        mixed_ce_ragged_offset + s, h, hd
                    ]
                    try:
                        assert_almost_equal(
                            true_ce_val,
                            mixed_ce_val,
                        )
                    except e:
                        print(
                            "MISMATCH:",
                            bs,
                            s,
                            h,
                            hd,
                        )
                        raise e

    _ = true_ce_q_ragged_host^
    _ = true_ce_q_ragged_device^
    _ = mixed_ce_q_ragged_host^
    _ = mixed_ce_q_ragged_device^
    _ = scale_host^
    _ = scale_device^
    _ = kv_block_paged_host^
    _ = kv_block_paged_device^
    _ = paged_lut_host^
    _ = paged_lut_device^
    _ = true_ce_output_device^
    _ = true_ce_output_host^
    _ = mixed_ce_output_device^
    _ = mixed_ce_output_host^
    _ = true_ce_cache_lengths_host^
    _ = true_ce_cache_lengths_device^
    _ = mixed_ce_cache_lengths_host^
    _ = mixed_ce_cache_lengths_device^
    _ = paged_lut_host^
    _ = paged_lut_device^


def main():
    with DeviceContext() as ctx:
        execute_ragged_flash_attention(ctx)
