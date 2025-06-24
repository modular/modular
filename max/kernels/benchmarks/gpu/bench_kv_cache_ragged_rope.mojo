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
from random import random_ui64, seed
from sys import env_get_dtype, env_get_int

from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import Dim, DimList
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer, arg_parse, random
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
)
from nn.fused_qk_rope import fused_qk_rope_ragged

from utils.index import IndexList


fn _get_run_name[
    type: DType,
    num_q_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
](batch_size: Int, seq_len: Int, use_random_seq_lengths: Bool,) -> String:
    # fmt: off
    return String(
        "fused_qkv_ragged_rope(", type, ") : ",

        # head_info
        "num_q_heads=", num_q_heads, ", ",
        "num_kv_heads=", num_kv_heads, ", ",
        "head_dim=", head_dim, " : ",

        "batch_size=", batch_size, ", ",
        "seq_len=", seq_len, ", ",
        "use_random_seq_lengths=", use_random_seq_lengths, ", ",
    )
    # fmt: on


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
    var input_row_offsets_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size + 1)
    )
    var cache_lengths_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size)
    )
    var max_prompt_length = 0
    var total_seq_len: UInt32 = 0
    var cache_len: UInt32 = 10

    var flop_count = 0
    for i in range(batch_size):
        var curr_seq_length: UInt32
        if use_random_seq_lengths:
            curr_seq_length = random_ui64(1, seq_len).cast[DType.uint32]()
        else:
            curr_seq_length = seq_len

        input_row_offsets_host.tensor[i] = curr_seq_length
        if curr_seq_length > max_prompt_length:
            max_prompt_length = Int(curr_seq_length)

        cache_lengths_host.tensor[i] = cache_len
        total_seq_len += curr_seq_length

    max_context_length = max_prompt_length + cache_len

    input_row_offsets_host.tensor[batch_size] = total_seq_len
    var input_row_offsets_device = input_row_offsets_host.copy_to_device(ctx)
    var cache_lengths_device = cache_lengths_host.copy_to_device(ctx)

    var q_host = HostNDBuffer[dtype, 3, DimList(Dim(), num_q_heads, head_dim)](
        IndexList[3](Int(total_seq_len), num_q_heads, head_dim)
    )
    random(q_host.tensor)
    var q_device = q_host.copy_to_device(ctx)
    var output_device = q_host.copy_to_device(ctx)

    var kv_block_device = DeviceNDBuffer[dtype, 6](
        IndexList[6](
            num_blocks,
            2,
            num_layers,
            Int(max_prompt_length + cache_len),
            num_kv_heads,
            head_dim,
        ),
        ctx=ctx,
    )

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
        max_context_length,
        max_context_length,
    )

    var freqs_cis_table_device = DeviceNDBuffer[
        dtype, 2, DimList(max_seq_len, head_dim)
    ]((max_seq_len, head_dim), ctx=ctx)

    num_flops_per_elem = 6
    num_elems = Int(total_seq_len) * num_q_heads * num_kv_heads * head_dim // 2
    flop_count = num_flops_per_elem * num_elems

    @parameter
    @__copy_capture(
        q_device,
        kv_collection_device,
        input_row_offsets_device,
        freqs_cis_table_device,
        output_device,
    )
    @always_inline
    fn bench_func(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            fused_qk_rope_ragged[
                CollectionType.CacheType,
                interleaved=False,
                target="gpu",
            ](
                q_device.tensor,
                input_row_offsets_device.tensor,
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
    _ = input_row_offsets_device^
    _ = cache_lengths_device^
    _ = lookup_table_device^
    _ = freqs_cis_table_device^


def main():
    alias dtype = env_get_dtype["dtype", DType.bfloat16]()

    alias head_dim = env_get_int["head_dim", 128]()
    alias num_q_heads = env_get_int["num_q_heads", 32]()
    alias num_kv_heads = env_get_int["num_kv_heads", 8]()

    var batch_size = arg_parse("batch_size", 1)
    var use_random_seq_lengths = arg_parse("use_random_lengths", False)
    var seq_len = arg_parse("seq_len", 1)

    seed(0)

    var m = Bench()
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

    m.dump_report()
