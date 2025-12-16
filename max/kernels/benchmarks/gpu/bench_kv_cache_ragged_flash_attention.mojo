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

from buffer import NDBuffer
from collections import Set, OptionalReg
from math import rsqrt
from random import random_ui64, seed
from sys import env_get_dtype, env_get_int

from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from gpu.host import DeviceContext
from internal_utils import arg_parse
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
)
from memory import UnsafePointer
from layout import UNKNOWN_VALUE, LayoutTensor, Layout, RuntimeLayout
from layout._fillers import random
from layout.layout import *
from nn.mha import flash_attention, flash_attention_ragged
from nn.mha_mask import CausalMask
from nn.mha_score_mod import IdentityScoreMod

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
    comptime num_layers = 1
    comptime layer_idx = 0

    var num_blocks = batch_size * 2
    comptime CollectionType = ContinuousBatchingKVCacheCollection[
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

    # Layouts for 1D buffers
    comptime row_offsets_layout = Layout.row_major(UNKNOWN_VALUE)
    var row_offsets_shape = IndexList[1](batch_size + 1)
    var row_offsets_runtime = RuntimeLayout[row_offsets_layout].row_major(
        row_offsets_shape
    )

    comptime cache_lengths_layout = Layout.row_major(UNKNOWN_VALUE)
    var cache_lengths_shape = IndexList[1](batch_size)
    var cache_lengths_runtime = RuntimeLayout[cache_lengths_layout].row_major(
        cache_lengths_shape
    )

    # Create device buffers for row offsets and cache lengths
    var input_row_offsets_device = ctx.enqueue_create_buffer[DType.uint32](
        row_offsets_shape.flattened_length()
    )
    var cache_lengths_device = ctx.enqueue_create_buffer[DType.uint32](
        cache_lengths_shape.flattened_length()
    )

    var max_context_length: UInt32 = 0
    var max_seq_length: UInt32 = 0
    var total_seq_len: UInt32 = 0
    var flop_count = 0

    # Initialize row offsets and cache lengths on host
    with input_row_offsets_device.map_to_host() as row_offsets_host:
        with cache_lengths_device.map_to_host() as cache_lengths_host:
            for i in range(batch_size):
                var curr_seq_length: UInt32
                if use_random_seq_lengths:
                    curr_seq_length = random_ui64(1, seq_len).cast[
                        DType.uint32
                    ]()
                else:
                    curr_seq_length = seq_len

                var curr_cache_length: UInt32
                if use_random_cache_lengths:
                    curr_cache_length = random_ui64(1, cache_len).cast[
                        DType.uint32
                    ]()
                else:
                    curr_cache_length = cache_len

                max_context_length = max(
                    max_context_length, curr_cache_length + curr_seq_length
                )
                max_seq_length = max(max_seq_length, curr_seq_length)

                row_offsets_host[i] = total_seq_len
                cache_lengths_host[i] = curr_cache_length
                total_seq_len += curr_seq_length

                flop_count += Int(
                    4
                    * num_q_heads
                    * (curr_cache_length + curr_seq_length)
                    * curr_seq_length
                    * head_dim
                )

            row_offsets_host[batch_size] = total_seq_len

    # Layout for Q tensor [total_seq_len, num_q_heads, head_dim]
    comptime q_layout = Layout.row_major(UNKNOWN_VALUE, num_q_heads, head_dim)
    var q_shape = IndexList[3](Int(total_seq_len), num_q_heads, head_dim)
    var q_runtime = RuntimeLayout[q_layout].row_major(q_shape)

    var q_device = ctx.enqueue_create_buffer[dtype](q_shape.flattened_length())

    # Initialize Q with random data
    with q_device.map_to_host() as q_host:
        var q_host_tensor = LayoutTensor[dtype, q_layout](q_host, q_runtime)
        random(q_host_tensor)

    # Create Q layout tensor
    var q_tensor = LayoutTensor[dtype, q_layout](q_device, q_runtime)

    # Output tensor [total_seq_len, num_q_heads, head_dim]
    comptime output_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, head_dim
    )
    var output_shape = IndexList[3](Int(total_seq_len), num_q_heads, head_dim)
    var output_runtime = RuntimeLayout[output_layout].row_major(output_shape)

    var output_device = ctx.enqueue_create_buffer[dtype](
        output_shape.flattened_length()
    )
    var output_device_tensor = LayoutTensor[dtype, output_layout](
        output_device, output_runtime
    )

    # KV block tensor [num_blocks, 2, num_layers, seq_len+cache_len, num_kv_heads, head_dim]
    comptime kv_block_layout = Layout.row_major[6]()
    var kv_block_shape = IndexList[6](
        num_blocks,
        2,
        num_layers,
        seq_len + cache_len,
        num_kv_heads,
        head_dim,
    )
    var kv_block_runtime = RuntimeLayout[kv_block_layout].row_major(
        kv_block_shape
    )

    var kv_block_device = ctx.enqueue_create_buffer[dtype](
        kv_block_shape.flattened_length()
    )

    # Initialize KV block with random data
    with kv_block_device.map_to_host() as kv_block_host:
        var kv_block_host_tensor = LayoutTensor[dtype, kv_block_layout](
            kv_block_host, kv_block_runtime
        )
        random(kv_block_host_tensor)

    var kv_block_tensor = LayoutTensor[dtype, kv_block_layout](
        kv_block_device, kv_block_runtime
    )

    # Lookup table [batch_size]
    comptime lookup_layout = Layout.row_major(UNKNOWN_VALUE)
    var lookup_shape = IndexList[1](batch_size)
    var lookup_runtime = RuntimeLayout[lookup_layout].row_major(lookup_shape)

    var lookup_table_device = ctx.enqueue_create_buffer[DType.uint32](
        lookup_shape.flattened_length()
    )

    # Initialize lookup table with random block indices
    with lookup_table_device.map_to_host() as lookup_host:
        var block_idx_set = Set[Int]()
        var idx = 0
        while idx < batch_size:
            var randval = Int(random_ui64(0, num_blocks - 1))
            if randval in block_idx_set:
                continue

            block_idx_set.add(randval)
            lookup_host[idx] = UInt32(randval)
            idx += 1

    # Create layout tensors for row offsets, cache lengths, and lookup table
    var input_row_offsets_tensor = LayoutTensor[
        DType.uint32, row_offsets_layout
    ](input_row_offsets_device, row_offsets_runtime)
    var cache_lengths_tensor = LayoutTensor[DType.uint32, cache_lengths_layout](
        cache_lengths_device, cache_lengths_runtime
    )
    var lookup_table_tensor = LayoutTensor[DType.uint32, lookup_layout](
        lookup_table_device, lookup_runtime
    )

    var kv_collection_device = CollectionType(
        LayoutTensor[dtype, Layout.row_major[6](), MutAnyOrigin](
            kv_block_tensor.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                kv_block_tensor.runtime_layout.shape.value,
                kv_block_tensor.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin](
            cache_lengths_tensor.ptr,
            RuntimeLayout[Layout(UNKNOWN_VALUE)](
                cache_lengths_tensor.runtime_layout.shape.value,
                cache_lengths_tensor.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin](
            lookup_table_tensor.ptr,
            RuntimeLayout[Layout(UNKNOWN_VALUE)](
                lookup_table_tensor.runtime_layout.shape.value,
                lookup_table_tensor.runtime_layout.stride.value,
            ),
        ),
        max_seq_length,
        max_context_length,
    )

    var k_cache_device = kv_collection_device.get_key_cache(layer_idx)
    var v_cache_device = kv_collection_device.get_value_cache(layer_idx)

    @parameter
    @__copy_capture(
        q_tensor,
        k_cache_device,
        v_cache_device,
        output_device_tensor,
        input_row_offsets_tensor,
    )
    @always_inline
    fn bench_func(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            flash_attention[ragged=True](
                output_device_tensor.as_any_origin(),
                q_tensor,
                k_cache_device,
                v_cache_device,
                CausalMask(),
                IdentityScoreMod(),
                input_row_offsets_tensor,
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
        [ThroughputMeasure(BenchMetric.flops, flop_count)],
    )


def main():
    comptime dtype = env_get_dtype["dtype", DType.bfloat16]()

    comptime head_dim = env_get_int["head_dim", 128]()
    comptime num_q_heads = env_get_int["num_q_heads", 32]()
    comptime num_kv_heads = env_get_int["num_kv_heads", 8]()

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
