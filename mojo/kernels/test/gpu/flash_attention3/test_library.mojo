# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: AMD-GPU
# REQUIRES: H100-GPU
# RUN: %mojo-no-debug-no-assert %s

from memory import memcpy
from flash_attention3.flash_attention import (
    daolabs_flash_attention3_paged_ragged_dispatch,
)
from nn.mha import flash_attention
from nn.mha_mask import CausalMask
from nn.mha_score_mod import IdentityScoreMod
from testing import assert_almost_equal
from gpu.host import DeviceContext
from random import seed, random_ui64
from memory import UnsafePointer
from math import ceildiv, isqrt
from collections import Set
from buffer import NDBuffer, Dim, DimList
from gpu.host._nvidia_cuda import CUDA
from kv_cache.types import PagedKVCache, KVCacheStaticParams
from internal_utils import HostNDBuffer, DeviceNDBuffer, random
from utils import IndexList
from utils.index import StaticTuple


def test_flash_attention[
    num_q_heads: Int, kv_params: KVCacheStaticParams, page_size: Int
](
    num_layers: Int,
    layer_idx: Int,
    valid_lengths: List[Int],
    cache_lengths: List[Int],
    ctx: DeviceContext,
):
    alias type = DType.bfloat16
    alias num_paged_blocks = 4096
    var flattened_num_paged_blocks = num_paged_blocks * num_layers
    var num_splits = 1
    alias PagedCacheType = PagedKVCache[type, kv_params, page_size]

    var batch_size = len(valid_lengths)
    var input_row_offsets_host = HostNDBuffer[DType.int32, 1](
        IndexList[1](batch_size + 1)
    )
    var input_row_offsets_ui32_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size + 1)
    )
    var total_length = 0
    var max_full_context_length = 0
    var max_prompt_length = 0
    var context_lengths_host = HostNDBuffer[DType.int32, 1](
        IndexList[1](batch_size)
    )
    var context_lengths_ui32_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size)
    )
    var cache_lengths_ui32_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size)
    )
    var cache_lengths_host = HostNDBuffer[DType.int32, 1](
        IndexList[1](batch_size)
    )
    var total_context_length = 0
    for i in range(batch_size):
        input_row_offsets_host.tensor[i] = total_length
        input_row_offsets_ui32_host.tensor[i] = total_length
        max_full_context_length = max(
            max_full_context_length, cache_lengths[i] + valid_lengths[i]
        )
        max_prompt_length = max(max_prompt_length, valid_lengths[i])
        total_length += valid_lengths[i]
        total_context_length += cache_lengths[i] + valid_lengths[i]
        context_lengths_host.tensor[i] = cache_lengths[i] + valid_lengths[i]
        context_lengths_ui32_host.tensor[i] = (
            cache_lengths[i] + valid_lengths[i]
        )
        cache_lengths_host.tensor[i] = cache_lengths[i]
        cache_lengths_ui32_host.tensor[i] = cache_lengths[i]
    input_row_offsets_host.tensor[batch_size] = total_length
    input_row_offsets_ui32_host.tensor[batch_size] = total_length
    context_lengths_device = context_lengths_host.copy_to_device(ctx)
    context_lengths_ui32_device = context_lengths_ui32_host.copy_to_device(ctx)
    cache_lengths_ui32_device = cache_lengths_ui32_host.copy_to_device(ctx)
    input_row_offsets_device = input_row_offsets_host.copy_to_device(ctx)
    input_row_offsets_ui32_device = input_row_offsets_ui32_host.copy_to_device(
        ctx
    )
    q_host = HostNDBuffer[
        type,
        3,
        DimList(Dim(), num_q_heads, kv_params.head_size),
    ](IndexList[3](total_length, num_q_heads, kv_params.head_size))
    random(q_host.tensor)
    # q_host.tensor.fill(BFloat16(0.5))
    q_device = q_host.copy_to_device(ctx)

    # initialize mask tensor
    # dummy mask to satisfy the argument.
    dummy_mask = NDBuffer[type, 4](
        UnsafePointer[Scalar[type]](), IndexList[4]()
    )

    # initialize reference output
    test_output_host = HostNDBuffer[
        type,
        3,
        DimList(Dim(), num_q_heads, kv_params.head_size),
    ](IndexList[3](total_length, num_q_heads, kv_params.head_size))
    test_output_device = test_output_host.copy_to_device(ctx)
    ref_output_host = HostNDBuffer[
        type,
        3,
        DimList(Dim(), num_q_heads, kv_params.head_size),
    ](IndexList[3](total_length, num_q_heads, kv_params.head_size))
    ref_output_device = ref_output_host.copy_to_device(ctx)

    their_kv_block_paged_host = HostNDBuffer[type, 5](
        IndexList[5](
            2,
            flattened_num_paged_blocks,
            page_size,
            kv_params.num_heads,
            kv_params.head_size,
        )
    )
    random(their_kv_block_paged_host.tensor)
    our_kv_block_paged_host = HostNDBuffer[type, 6](
        IndexList[6](
            num_paged_blocks,
            2,
            num_layers,
            page_size,
            kv_params.num_heads,
            kv_params.head_size,
        )
    )

    their_paged_lut_host = HostNDBuffer[DType.int32, 2](
        IndexList[2](batch_size, ceildiv(max_full_context_length, page_size))
    )
    our_paged_lut_host = HostNDBuffer[DType.uint32, 2](
        IndexList[2](batch_size, ceildiv(max_full_context_length, page_size))
    )
    paged_lut_set = Set[Int]()
    for bs in range(batch_size):
        seq_len = cache_lengths[bs] + valid_lengths[bs]
        for block_idx in range(0, ceildiv(seq_len, page_size)):
            if len(paged_lut_set) == num_paged_blocks:
                raise "Not enough paged blocks to handle request"
            var randval = Int(random_ui64(0, num_paged_blocks - 1))
            while randval in paged_lut_set:
                randval = Int(random_ui64(0, num_paged_blocks - 1))

            paged_lut_set.add(randval)
            our_paged_lut_host.tensor[bs, block_idx] = randval
            their_randval = randval * num_layers + layer_idx
            their_paged_lut_host.tensor[bs, block_idx] = their_randval
            memcpy(
                our_kv_block_paged_host.tensor._offset(
                    StaticTuple[Int, 6](randval, 0, layer_idx, 0, 0, 0)
                ),
                their_kv_block_paged_host.tensor._offset(
                    StaticTuple[Int, 5](
                        0,
                        their_randval,
                        0,
                        0,
                        0,
                    )
                ),
                page_size * kv_params.num_heads * kv_params.head_size,
            )
            memcpy(
                our_kv_block_paged_host.tensor._offset(
                    StaticTuple[Int, 6](randval, 1, layer_idx, 0, 0, 0)
                ),
                their_kv_block_paged_host.tensor._offset(
                    StaticTuple[Int, 5](
                        1,
                        their_randval,
                        0,
                        0,
                        0,
                    )
                ),
                page_size * kv_params.num_heads * kv_params.head_size,
            )

    their_paged_lut_device = their_paged_lut_host.copy_to_device(ctx)
    our_paged_lut_device = our_paged_lut_host.copy_to_device(ctx)
    their_kv_block_paged_device = their_kv_block_paged_host.copy_to_device(ctx)
    our_kv_block_paged_device = our_kv_block_paged_host.copy_to_device(ctx)

    daolabs_flash_attention3_paged_ragged_dispatch[type, kv_params, page_size](
        q_device.tensor,
        their_kv_block_paged_device.tensor,
        input_row_offsets_device.tensor,
        context_lengths_device.tensor,
        their_paged_lut_device.tensor,
        test_output_device.tensor,
        max_prompt_length,
        max_full_context_length,
        ctx,
    )

    var k_cache_opaque = PagedCacheType(
        our_kv_block_paged_device.tensor,
        cache_lengths_ui32_device.tensor,
        our_paged_lut_device.tensor,
        max_prompt_length,
        max_full_context_length,
        layer_idx,
        PagedCacheType.KeyIdx,
    )
    var v_cache_opaque = PagedCacheType(
        our_kv_block_paged_device.tensor,
        cache_lengths_ui32_device.tensor,
        our_paged_lut_device.tensor,
        max_prompt_length,
        max_full_context_length,
        layer_idx,
        PagedCacheType.ValueIdx,
    )
    flash_attention[add_attn_mask=False, ragged=True](
        ref_output_device.tensor,
        q_device.tensor,
        k_cache_opaque,
        v_cache_opaque,
        dummy_mask,
        CausalMask(),
        IdentityScoreMod(),
        input_row_offsets_ui32_device.tensor,
        isqrt(Float32(kv_params.head_size)),
        ctx,
    )
    ctx.enqueue_copy(test_output_host.tensor.data, test_output_device.buffer)
    ctx.enqueue_copy(ref_output_host.tensor.data, ref_output_device.buffer)
    ctx.synchronize()

    for i in range(batch_size):
        seq_len = valid_lengths[i]
        start_offset = input_row_offsets_host.tensor[i]
        for s in range(seq_len):
            for j in range(num_q_heads):
                for k in range(kv_params.head_size):
                    try:
                        assert_almost_equal(
                            ref_output_host.tensor[Int(start_offset + s), j, k],
                            test_output_host.tensor[
                                Int(start_offset + s), j, k
                            ],
                            atol=4e-3,  # threshold is the same for what they use in their test.
                            rtol=1e-5,
                        )
                    except e:
                        print(
                            "Error at",
                            i,
                            s,
                            j,
                            k,
                            ref_output_host.tensor[Int(start_offset + s), j, k],
                            test_output_host.tensor[
                                Int(start_offset + s), j, k
                            ],
                        )
                        raise e

    _ = ref_output_device^
    _ = q_device^
    _ = their_kv_block_paged_device^
    _ = our_paged_lut_device^
    _ = input_row_offsets_ui32_device^
    _ = context_lengths_ui32_device^
    _ = ref_output_host^
    _ = test_output_host^


def test_flash_attention_suite(ctx: DeviceContext):
    # context encoding
    alias ce_num_q_heads = 32
    alias ce_kv_params = KVCacheStaticParams(num_heads=8, head_size=128)
    alias ce_page_size = 128
    var ce_valid_lengths = List[Int](500, 200, 200, 200)
    var ce_cache_lengths = List[Int](0, 0, 0, 0)
    var ce_num_layers = 1
    var ce_layer_idx = 0
    print("running CE with single layer")
    test_flash_attention[ce_num_q_heads, ce_kv_params, ce_page_size](
        ce_num_layers, ce_layer_idx, ce_valid_lengths, ce_cache_lengths, ctx
    )

    # context encoding more layers
    ce_num_layers = 2
    ce_layer_idx = 1
    print("running CE with two layers")
    test_flash_attention[ce_num_q_heads, ce_kv_params, ce_page_size](
        ce_num_layers, ce_layer_idx, ce_valid_lengths, ce_cache_lengths, ctx
    )

    # token gen
    alias tg_num_q_heads = 32
    alias tg_kv_params = KVCacheStaticParams(num_heads=8, head_size=128)
    alias tg_page_size = 128
    var tg_valid_lengths = List[Int](1, 1, 1, 1)
    var tg_cache_lengths = List[Int](128, 256, 300, 500)
    var tg_num_layers = 1
    var tg_layer_idx = 0
    print("running TG with single layer")
    test_flash_attention[tg_num_q_heads, tg_kv_params, tg_page_size](
        tg_num_layers, tg_layer_idx, tg_valid_lengths, tg_cache_lengths, ctx
    )

    # token gen more layers
    tg_num_layers = 2
    tg_layer_idx = 1
    print("running TG with two layers")
    test_flash_attention[tg_num_q_heads, tg_kv_params, tg_page_size](
        tg_num_layers, tg_layer_idx, tg_valid_lengths, tg_cache_lengths, ctx
    )


def main():
    with DeviceContext() as ctx:
        test_flash_attention_suite(ctx)
