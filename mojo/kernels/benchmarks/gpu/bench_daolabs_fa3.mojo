# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: AMD-GPU
# REQUIRES: H100-GPU
# RUN: %mojo-build-no-debug-no-assert %s

from memory import memcpy
from flash_attention3.flash_attention import (
    daolabs_flash_attention3_paged_ragged_dispatch,
)
from nn.mha import flash_attention
from nn.mha_mask import CausalMask
from nn.mha_score_mod import IdentityScoreMod
from testing import assert_almost_equal
from gpu.host import DeviceContext
from internal_utils import arg_parse
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
from sys import env_get_bool, env_get_dtype, env_get_int, sizeof
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure


def flops(
    batch: Int, nheads: Int, seqlen_q: Int, seqlen_k: Int, headdim: Int
) -> Int:
    var avg_seqlen = (max(0, seqlen_k - seqlen_q) + seqlen_k) / 2
    return Int(batch * nheads * 2 * seqlen_q * avg_seqlen * (headdim + headdim))


fn _get_run_name[
    type: DType,
    num_q_heads: Int,
    num_kv_heads: Int,
    page_size: Int,
    head_dim: Int,
](batch_size: Int, seq_len: Int, cache_len: Int,) -> String:
    var name = String(
        "daolabs_flash_attention3",
        "(",
        type,
        ") : ",
        # head_info
        "num_q_heads=",
        num_q_heads,
        ", num_kv_heads=",
        num_kv_heads,
        ", head_dim=",
        head_dim,
        " : ",
        "batch_size=",
        batch_size,
        ", seq_len=",
        seq_len,
        ", cache_len=",
        cache_len,
        ", page_size=",
        page_size,
    )

    return name


def test_flash_attention[
    num_q_heads: Int, kv_params: KVCacheStaticParams, page_size: Int
](
    mut m: Bench,
    batch_size: Int,
    prompt_length: Int,
    cache_length: Int,
    ctx: DeviceContext,
):
    alias type = DType.bfloat16

    alias PagedCacheType = PagedKVCache[type, kv_params, page_size]

    var num_layers = 1
    layer_idx = 0
    var num_paged_blocks = Int(
        ceildiv(batch_size * (prompt_length + cache_length), page_size) * 1.2
    )
    var flattened_num_paged_blocks = num_paged_blocks * num_layers
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
            max_full_context_length, cache_length + prompt_length
        )
        max_prompt_length = max(max_prompt_length, prompt_length)
        total_length += prompt_length
        total_context_length += cache_length + prompt_length
        context_lengths_host.tensor[i] = cache_length + prompt_length
        context_lengths_ui32_host.tensor[i] = cache_length + prompt_length
        cache_lengths_host.tensor[i] = cache_length
        cache_lengths_ui32_host.tensor[i] = cache_length

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
    paged_lut_set = Set[Int]()
    for bs in range(batch_size):
        seq_len = cache_length + prompt_length
        for block_idx in range(0, ceildiv(seq_len, page_size)):
            if len(paged_lut_set) == num_paged_blocks:
                raise "Not enough paged blocks to handle request"
            var randval = Int(random_ui64(0, num_paged_blocks))
            while randval in paged_lut_set:
                randval = Int(random_ui64(0, num_paged_blocks))

            paged_lut_set.add(randval)
            their_randval = randval * num_layers + layer_idx
            their_paged_lut_host.tensor[bs, block_idx] = their_randval

    their_paged_lut_device = their_paged_lut_host.copy_to_device(ctx)
    their_kv_block_paged_device = their_kv_block_paged_host.copy_to_device(ctx)

    @parameter
    @__copy_capture(
        q_device,
        their_kv_block_paged_device,
        input_row_offsets_device,
        context_lengths_device,
        their_paged_lut_device,
        test_output_device,
    )
    @always_inline
    fn bench_func(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            daolabs_flash_attention3_paged_ragged_dispatch[
                type, kv_params, page_size
            ](
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

        b.iter_custom[kernel_launch](ctx)

    flop_count = flops(
        batch_size,
        num_q_heads,
        prompt_length,
        cache_length + prompt_length,
        kv_params.head_size,
    )
    m.bench_function[bench_func](
        BenchId(
            _get_run_name[
                type,
                num_q_heads,
                kv_params.num_heads,
                page_size,
                kv_params.head_size,
            ](
                batch_size,
                prompt_length,
                cache_length,
            )
        ),
        ThroughputMeasure(BenchMetric.flops, flop_count),
    )

    _ = ref_output_device^
    _ = q_device^
    _ = their_kv_block_paged_device^
    _ = input_row_offsets_ui32_device^
    _ = context_lengths_ui32_device^
    _ = ref_output_host^
    _ = test_output_host^


def main():
    alias dtype = DType.bfloat16

    alias head_dim = env_get_int["head_dim", 128]()
    alias num_q_heads = env_get_int["num_q_heads", 32]()
    alias num_kv_heads = env_get_int["num_kv_heads", 8]()

    var batch_size = arg_parse("batch_size", 1)
    var use_random_seq_lengths = arg_parse("use_random_seq_lengths", False)
    var seq_len = arg_parse("seq_len", 1)
    var cache_len = arg_parse("cache_len", 1)

    var m = Bench()
    with DeviceContext() as ctx:
        test_flash_attention[
            num_q_heads,
            KVCacheStaticParams(num_heads=num_kv_heads, head_size=head_dim),
            128,
        ](m, batch_size, seq_len, cache_len, ctx)

    m.dump_report()
