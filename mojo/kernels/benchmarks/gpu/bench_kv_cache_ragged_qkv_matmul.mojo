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
from kv_cache.types import ContinuousBatchingKVCache, KVCacheStaticParams
from collections import Set
from memory import UnsafePointer
from nn.kv_cache_ragged import _fused_qkv_matmul_kv_cache_ragged_impl


fn _get_run_name[
    type: DType,
    num_q_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
](seq_len: Int, batch_size: Int, use_random_lengths: Bool) -> String:
    var name = String("fused_qkv_ragged_matmul") + "("
    name += str(type)
    name += ") : "

    # head_info
    name += "num_q_heads=" + str(num_q_heads) + ", "
    name += "num_kv_heads=" + str(num_kv_heads) + ", "
    name += "head_dim=" + str(head_dim) + " :"

    name += "batch_size=" + str(batch_size) + ", "
    name += "seq_len=" + str(seq_len)
    name += "use_random_lengths=" + str(use_random_lengths)

    return name


def execute_kv_cache_ragged_matmul[
    dtype: DType, head_dim: Int, num_q_heads: Int, num_kv_heads: Int
](
    ctx: DeviceContext,
    mut m: Bench,
    batch_size: Int,
    seq_len: Int,
    use_random_lengths: Bool,
):
    alias CacheType = ContinuousBatchingKVCache[
        dtype,
        KVCacheStaticParams(num_heads=num_kv_heads, head_size=head_dim),
    ]

    alias hidden_size = num_q_heads * head_dim
    alias combined_hidden_size = (num_q_heads + 2 * num_kv_heads) * head_dim
    var num_blocks = batch_size + 1
    alias num_layers = 1
    alias max_seq_length_cache = 1024
    alias is_context_encoding = True  # value is ignored for matmul kernel
    alias layer_idx = 0

    var total_seq_len: UInt32 = 0
    var prefix_sums_host = HostNDBuffer[DType.uint32, 1](
        (batch_size + 1,),
    )
    for i in range(batch_size):
        var length: UInt32
        if use_random_lengths:
            length = random_ui64(1, seq_len).cast[DType.uint32]()
        else:
            length = seq_len

        prefix_sums_host.tensor[i] = length
        total_seq_len += length

    prefix_sums_host.tensor[batch_size] = total_seq_len
    var prefix_sums_device = prefix_sums_host.copy_to_device(ctx)

    var hidden_state_host = HostNDBuffer[dtype, 2, DimList(Dim(), hidden_size)](
        (int(total_seq_len), hidden_size),
    )
    random(hidden_state_host.tensor)
    var hidden_state_device = hidden_state_host.copy_to_device(ctx)

    var weight_host = HostNDBuffer[
        dtype, 2, DimList(hidden_size, combined_hidden_size)
    ]((hidden_size, combined_hidden_size))
    random(weight_host.tensor)
    var weight_device = weight_host.copy_to_device(ctx)

    var output_host = HostNDBuffer[dtype, 2, DimList(Dim(), hidden_size)](
        (int(total_seq_len), combined_hidden_size),
    )
    random(output_host.tensor)
    var output_device = output_host.copy_to_device(ctx)

    var kv_block_host = HostNDBuffer[dtype, 6,](
        IndexList[6](
            num_blocks,
            2,
            num_layers,
            max_seq_length_cache,
            num_kv_heads,
            head_dim,
        ),
    )
    var kv_block_device = kv_block_host.copy_to_device(ctx)
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

    # initialize our KVCache
    var cache_lengths_host = HostNDBuffer[DType.uint32, 1]((batch_size,))
    for i in range(batch_size):
        cache_lengths_host.tensor[i] = 10

    var cache_lengths_device = cache_lengths_host.copy_to_device(ctx)

    var k_cache_device = CacheType(
        kv_block_device.tensor,
        cache_lengths_device.tensor,
        lookup_table_device.tensor,
        is_context_encoding,
        layer_idx,
        CacheType.KeyIdx,
    )
    var v_cache_device = CacheType(
        kv_block_device.tensor,
        cache_lengths_device.tensor,
        lookup_table_device.tensor,
        is_context_encoding,
        layer_idx,
        CacheType.ValueIdx,
    )

    @parameter
    @__copy_capture(
        hidden_state_device,
        prefix_sums_device,
        k_cache_device,
        v_cache_device,
        output_device,
    )
    @always_inline
    fn bench_func(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            _fused_qkv_matmul_kv_cache_ragged_impl[target="cuda",](
                hidden_state_device.tensor,
                prefix_sums_device.tensor,
                weight_device.tensor,
                k_cache_device,
                v_cache_device,
                output_device.tensor,
                ctx,
            )

        b.iter_custom[kernel_launch](ctx)

    m.bench_function[bench_func](
        BenchId(
            _get_run_name[dtype, num_q_heads, num_kv_heads, head_dim](
                seq_len,
                batch_size,
                use_random_lengths,
            )
        ),
        # TODO: Pick relevant benchmetric
        ThroughputMeasure(
            BenchMetric.flops,
            # Flop: 2*M*N*K. Use A and C shapes since they're not transposed.
            2 * int(total_seq_len) * hidden_size * combined_hidden_size,
        ),
    )


def main():
    alias dtype = env_get_dtype["dtype", DType.bfloat16]()

    alias batch_size = env_get_int["batch_size", 1]()
    alias use_random_lengths = env_get_bool["use_random_lengths", False]()
    alias seq_len = env_get_int["seq_len", 1]()
    alias head_dim = env_get_int["head_dim", 128]()
    alias num_q_heads = env_get_int["num_q_heads", 128]()
    alias num_kv_heads = env_get_int["num_kv_heads", 128]()

    seed(0)

    var m = Bench()
    try:
        with DeviceContext() as ctx:
            # benchmarking matmul
            execute_kv_cache_ragged_matmul[
                dtype,
                head_dim,
                num_q_heads,
                num_kv_heads,
            ](
                ctx,
                m,
                batch_size,
                seq_len,
                use_random_lengths,
            )

    except e:
        print("CUDA_ERROR:", e)

    m.dump_report()
