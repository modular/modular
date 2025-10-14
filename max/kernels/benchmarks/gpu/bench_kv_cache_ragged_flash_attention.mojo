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
from math import rsqrt
from random import random_ui64, seed
from sys import env_get_dtype, env_get_int

from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext
from internal_utils import HostNDBuffer, arg_parse, random
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
)
from nn.mha import flash_attention
from nn.mha_mask import CausalMask
from nn.mha_score_mod import IdentityScoreMod
from tensor_internal import IOUnknown, ManagedTensorSlice
from tensor_internal.managed_tensor_slice import StaticTensorSpec

from utils import IndexList


fn _get_run_name[
    dtype: DType,
    num_q_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
](
    batch_size: Int,
    seq_len: Int,
    use_random_seq_lengths: Bool,
    cache_len: Int,
    use_random_cache_lengths: Bool,
) -> String:
    # fmt: off
    return String(
        "fused_qkv_ragged_flash_attention(", dtype, ") : "

        # head_info
        "num_q_heads=", num_q_heads, ", ",
        "num_kv_heads=", num_kv_heads, ", ",
        "head_dim=", head_dim, " : ",

        "batch_size=", batch_size, ", ",
        "seq_len=", seq_len, ", ",
        "use_random_seq_lengths=", use_random_seq_lengths, ", ",
        "cache_len=", cache_len, ", ",
        "use_random_cache_lengths=", use_random_cache_lengths
    )
    # fmt: on


def execute_kv_cache_ragged_flash_attention[
    dtype: DType,
    head_dim: Int,
    num_q_heads: Int,
    num_kv_heads: Int,
](
    ctx: DeviceContext,
    mut m: Bench,
    batch_size: Int,
    seq_len: Int,
    use_random_seq_lengths: Bool,
    cache_len: Int,
    use_random_cache_lengths: Bool,
):
    alias num_layers = 1
    alias layer_idx = 0

    var num_blocks = batch_size * 2
    alias CollectionType = ContinuousBatchingKVCacheCollection[
        dtype,
        KVCacheStaticParams(
            num_heads=UInt(num_kv_heads), head_size=UInt(head_dim)
        ),
    ]

    debug_assert(
        batch_size < num_blocks,
        "batch_size passed to unit test (",
        batch_size,
        ") is larger than configured num_blocks (",
        num_blocks,
        ")",
    )

    var input_row_offsets_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size + 1)
    )
    var cache_lengths_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size)
    )
    var max_context_length: UInt32 = 0
    var max_seq_length: UInt32 = 0
    var total_seq_len: UInt32 = 0

    var flop_count = 0
    for i in range(batch_size):
        var curr_seq_length: UInt32
        if use_random_seq_lengths:
            curr_seq_length = random_ui64(1, seq_len).cast[DType.uint32]()
        else:
            curr_seq_length = seq_len

        var curr_cache_length: UInt32
        if use_random_cache_lengths:
            curr_cache_length = random_ui64(1, cache_len).cast[DType.uint32]()
        else:
            curr_cache_length = cache_len

        max_context_length = max(
            max_context_length, curr_cache_length + curr_seq_length
        )
        max_seq_length = max(max_seq_length, curr_seq_length)

        input_row_offsets_host.tensor[i] = total_seq_len
        cache_lengths_host.tensor[i] = curr_cache_length
        total_seq_len += curr_seq_length

        flop_count += Int(
            4
            * num_q_heads
            * (curr_cache_length + curr_seq_length)
            * curr_seq_length
            * head_dim
        )

    input_row_offsets_host.tensor[batch_size] = total_seq_len
    var input_row_offsets_device = input_row_offsets_host.copy_to_device(ctx)
    var cache_lengths_device = cache_lengths_host.copy_to_device(ctx)

    q_host = HostNDBuffer[dtype, 3, DimList(Dim(), num_q_heads, head_dim)](
        IndexList[3](Int(total_seq_len), num_q_heads, head_dim)
    )
    random(q_host.tensor)
    var q_device = q_host.copy_to_device(ctx)

    # initialize mask tensor
    # dummy mask to satisfy the argument.
    dummy_mask = NDBuffer[dtype, 4](
        UnsafePointer[Scalar[dtype]](), IndexList[4]()
    )

    # initialize reference output
    output_host = HostNDBuffer[dtype, 3, DimList(Dim(), num_q_heads, head_dim)](
        IndexList[3](Int(total_seq_len), num_q_heads, head_dim)
    )
    var output_device = output_host.copy_to_device(ctx)

    # initialize our KVCache
    kv_block_host = HostNDBuffer[dtype, 6](
        IndexList[6](
            num_blocks,
            2,
            num_layers,
            seq_len + cache_len,
            num_kv_heads,
            head_dim,
        ),
    )
    random(kv_block_host.tensor)
    var kv_block_device = kv_block_host.copy_to_device(ctx)

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
        max_seq_length,
        max_context_length,
    )

    var k_cache_device = kv_collection_device.get_key_cache(layer_idx)
    var v_cache_device = kv_collection_device.get_value_cache(layer_idx)

    @parameter
    @__copy_capture(
        q_device,
        k_cache_device,
        v_cache_device,
        output_device,
        dummy_mask,
        input_row_offsets_device,
    )
    @always_inline
    fn bench_func(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            flash_attention[ragged=True](
                # TODO: remove the origin cast once unified closures are supported.
                output_device.to_layout_tensor().origin_cast[True](),
                q_device.to_layout_tensor(),
                k_cache_device,
                v_cache_device,
                CausalMask(),
                IdentityScoreMod(),
                ManagedTensorSlice[
                    io_spec=IOUnknown,
                    static_spec = StaticTensorSpec[
                        DType.uint32, 1
                    ].create_unknown(),
                ](input_row_offsets_device.tensor),
                rsqrt(Float32(head_dim)),
                ctx,
            )

        b.iter_custom[kernel_launch](ctx)

    m.bench_function[bench_func](
        BenchId(
            _get_run_name[dtype, num_q_heads, num_kv_heads, head_dim](
                batch_size,
                seq_len,
                use_random_seq_lengths,
                cache_len,
                use_random_cache_lengths,
            )
        ),
        ThroughputMeasure(BenchMetric.flops, flop_count),
    )
    _ = kv_block_device^
    _ = output_device^
    _ = q_device^
    _ = input_row_offsets_device^
    _ = cache_lengths_device^
    _ = lookup_table_device^


def main():
    alias dtype = env_get_dtype["dtype", DType.bfloat16]()

    alias head_dim = env_get_int["head_dim", 128]()
    alias num_q_heads = env_get_int["num_q_heads", 32]()
    alias num_kv_heads = env_get_int["num_kv_heads", 8]()

    var batch_size = arg_parse("batch_size", 1)
    var use_random_seq_lengths = arg_parse("use_random_seq_lengths", False)
    var seq_len = arg_parse("seq_len", 1)
    var cache_len = arg_parse("cache_len", 1)
    var use_random_cache_lengths = arg_parse("use_random_cache_lengths", False)

    seed(0)

    var m = Bench()
    with DeviceContext() as ctx:
        # benchmarking flash attention
        execute_kv_cache_ragged_flash_attention[
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
            cache_len,
            use_random_cache_lengths,
        )

    m.dump_report()
