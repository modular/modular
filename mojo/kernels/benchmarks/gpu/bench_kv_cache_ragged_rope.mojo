# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-build-no-debug %s
from internal_utils import env_get_dtype, DeviceNDBuffer, HostNDBuffer, random
from random import random_ui64, seed
from sys import env_get_int, sizeof, env_get_bool
from gpu.host import DeviceBuffer, DeviceContext
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import Dim, DimList
from utils.index import IndexList
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
)
from collections import Set
from memory import UnsafePointer
from nn.fused_qk_rope import fused_qk_rope_ragged


fn _get_run_name[
    type: DType,
    num_q_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
](batch_size: Int, seq_len: Int, use_random_seq_lengths: Bool,) -> String:
    var name = String("fused_qkv_ragged_rope") + "("
    name += str(type)
    name += ") : "

    # head_info
    name += "num_q_heads=" + str(num_q_heads) + ", "
    name += "num_kv_heads=" + str(num_kv_heads) + ", "
    name += "head_dim=" + str(head_dim) + " : "

    name += "batch_size=" + str(batch_size) + ", "
    name += "seq_len=" + str(seq_len) + ", "
    name += "use_random_seq_lengths=" + str(use_random_seq_lengths) + ", "

    return name


def execute_kv_cache_ragged_rope[
    dtype: DType, head_dim: Int, num_q_heads: Int, num_kv_heads: Int
](
    ctx: DeviceContext,
    mut m: Bench,
    batch_size: Int,
    seq_len: Int,
    use_random_seq_lengths: Bool,
):
    alias max_seq_len = 2048
    var num_blocks = batch_size * 2
    var num_layers = 1

    alias CollectionType = ContinuousBatchingKVCacheCollection[
        dtype,
        KVCacheStaticParams(num_heads=num_kv_heads, head_size=head_dim),
    ]
    var input_row_offset_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size + 1)
    )
    var cache_lengths_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size)
    )
    var max_prompt_length = 0
    var is_context_encoding = False  # unused by rope
    var total_seq_len: UInt32 = 0
    var cache_len: UInt32 = 10

    var flop_count = 0
    var seq_ids = List[Int]()
    for i in range(batch_size):
        var curr_seq_length: UInt32
        if use_random_seq_lengths:
            curr_seq_length = random_ui64(1, seq_len).cast[DType.uint32]()
        else:
            curr_seq_length = seq_len

        input_row_offset_host.tensor[i] = curr_seq_length
        if curr_seq_length > max_prompt_length:
            max_prompt_length = int(curr_seq_length)

        cache_lengths_host.tensor[i] = cache_len
        total_seq_len += curr_seq_length
        seq_ids.append(-1)

    input_row_offset_host.tensor[batch_size] = total_seq_len
    var input_row_offset_device = input_row_offset_host.copy_to_device(ctx)
    var cache_lengths_device = cache_lengths_host.copy_to_device(ctx)

    var q_host = HostNDBuffer[dtype, 3, DimList(Dim(), num_q_heads, head_dim)](
        IndexList[3](int(total_seq_len), num_q_heads, head_dim)
    )
    random(q_host.tensor)
    var q_device = q_host.copy_to_device(ctx)
    var output_device = q_host.copy_to_device(ctx)

    var kv_block_device = DeviceNDBuffer[dtype, 6,](
        IndexList[6](
            num_blocks,
            2,
            num_layers,
            int(max_prompt_length + cache_len),
            num_kv_heads,
            head_dim,
        ),
        ctx=ctx,
    )

    var lookup_table_host = HostNDBuffer[DType.uint32, 1,](
        IndexList[1](
            batch_size,
        ),
    )

    # hacky way to select random blocks.
    var block_idx_set = Set[Int]()
    var idx = 0
    while idx < batch_size:
        var randval = int(random_ui64(0, num_blocks - 1))
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
        is_context_encoding,
        seq_ids,
    )

    var freqs_cis_table_device = DeviceNDBuffer[
        dtype, 2, DimList(max_seq_len, head_dim)
    ]((max_seq_len, head_dim), ctx=ctx)

    num_flops_per_elem = 6
    num_elems = int(total_seq_len) * num_q_heads * num_kv_heads * head_dim // 2
    flop_count = num_flops_per_elem * num_elems

    @parameter
    @__copy_capture(
        q_device,
        kv_collection_device,
        input_row_offset_device,
        freqs_cis_table_device,
        output_device,
    )
    @always_inline
    fn bench_func(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            fused_qk_rope_ragged[CollectionType.CacheType, target="gpu",](
                q_device.tensor,
                input_row_offset_device.tensor,
                kv_collection_device,
                freqs_cis_table_device.tensor,
                0,
                output_device.tensor,
                ctx,
            )

        b.iter_custom[kernel_launch](ctx)

    m.bench_function[bench_func](
        BenchId(
            _get_run_name[dtype, num_q_heads, num_kv_heads, head_dim](
                batch_size,
                seq_len,
                use_random_seq_lengths,
            )
        ),
        ThroughputMeasure(BenchMetric.flops, flop_count),
    )

    _ = kv_block_device^
    _ = output_device^
    _ = q_device^
    _ = input_row_offset_device^
    _ = cache_lengths_device^
    _ = lookup_table_device^
    _ = freqs_cis_table_device^


def main():
    alias dtype = env_get_dtype["dtype", DType.bfloat16]()

    alias batch_size = env_get_int["batch_size", 1]()
    alias use_random_seq_lengths = env_get_bool[
        "use_random_seq_lengths", False
    ]()
    alias seq_len = env_get_int["seq_len", 1]()
    alias head_dim = env_get_int["head_dim", 128]()
    alias num_q_heads = env_get_int["num_q_heads", 32]()
    alias num_kv_heads = env_get_int["num_kv_heads", 8]()

    seed(0)

    var m = Bench()
    try:
        with DeviceContext() as ctx:
            # benchmarking flash attention
            execute_kv_cache_ragged_rope[
                dtype,
                head_dim,
                num_q_heads,
                num_kv_heads,
            ](
                ctx,
                m,
                batch_size,
                seq_len,
                use_random_seq_lengths,
            )

    except e:
        print("CUDA_ERROR:", e)

    m.dump_report()
