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
from math import ceildiv
from random import random_ui64, seed
from sys import env_get_dtype, env_get_int

from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from gpu.host import DeviceContext
from internal_utils import arg_parse
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
)
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from nn.kv_cache_ragged import _fused_qkv_matmul_kv_cache_ragged_impl
from utils import IndexList


fn _get_run_name[
    dtype: DType,
    num_q_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
](seq_len: Int, batch_size: Int, use_random_lengths: Bool) -> String:
    # fmt: off
    return String(
        "fused_qkv_ragged_matmul(", dtype, ") : ",

        # head_info
        "num_q_heads=", num_q_heads, ", ",
        "num_kv_heads=", num_kv_heads, ", ",
        "head_dim=", head_dim, " :",

        "batch_size=", batch_size, ", ",
        "seq_len=", seq_len, ", ",
        "use_random_lengths=", use_random_lengths,
    )
    # fmt: on


def execute_kv_cache_ragged_matmul[
    dtype: DType, head_dim: Int, num_q_heads: Int, num_kv_heads: Int
](
    ctx: DeviceContext,
    mut m: Bench,
    batch_size: Int,
    seq_len: Int,
    use_random_lengths: Bool,
):
    comptime CollectionType = ContinuousBatchingKVCacheCollection[
        dtype,
        KVCacheStaticParams(
            num_heads=UInt(num_kv_heads), head_size=UInt(head_dim)
        ),
    ]

    comptime hidden_size = num_q_heads * head_dim
    comptime combined_hidden_size = (num_q_heads + 2 * num_kv_heads) * head_dim
    var num_blocks = batch_size + 1
    comptime max_seq_length_cache = 1024
    comptime num_layers = 1
    comptime cache_size = 10
    comptime is_context_encoding = True  # value is ignored for matmul kernel
    comptime layer_idx = 0

    var max_context_length = 0
    var max_prompt_length = 0
    var total_seq_len: UInt32 = 0
    var prefix_sums_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    var prefix_sums_device_tensor = LayoutTensor[
        DType.uint32, Layout(UNKNOWN_VALUE)
    ](
        prefix_sums_device,
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(
            IndexList[1](batch_size + 1)
        ),
    )

    with prefix_sums_device.map_to_host() as prefix_sums_host:
        for i in range(batch_size):
            var length: UInt32
            if use_random_lengths:
                length = random_ui64(1, seq_len).cast[DType.uint32]()
            else:
                length = seq_len

            prefix_sums_host[i] = length
            total_seq_len += length
            max_context_length = max(
                max_context_length, Int(length + cache_size)
            )
            max_prompt_length = max(max_prompt_length, Int(length))
        prefix_sums_host[batch_size] = total_seq_len
    # Hidden state tensor layout and buffer
    comptime hidden_state_static_shape = Layout.row_major(
        UNKNOWN_VALUE, hidden_size
    )
    var hidden_state_dynamic_shape = IndexList[2](
        Int(total_seq_len), hidden_size
    )
    var hidden_state_runtime_layout = RuntimeLayout[
        hidden_state_static_shape
    ].row_major(hidden_state_dynamic_shape)
    var hidden_state_buffer = ctx.enqueue_create_buffer[dtype](
        hidden_state_dynamic_shape.flattened_length()
    )
    var hidden_state_device = LayoutTensor[dtype, hidden_state_static_shape](
        hidden_state_buffer, hidden_state_runtime_layout
    )

    with hidden_state_buffer.map_to_host() as hidden_state_host:
        var hidden_state_host_tensor = LayoutTensor[
            dtype, hidden_state_static_shape
        ](hidden_state_host, hidden_state_runtime_layout)
        random(hidden_state_host_tensor)

    # Weight tensor layout and buffer
    comptime weight_static_shape = Layout.row_major(
        hidden_size, combined_hidden_size
    )
    var weight_dynamic_shape = IndexList[2](hidden_size, combined_hidden_size)
    var weight_runtime_layout = RuntimeLayout[weight_static_shape].row_major(
        weight_dynamic_shape
    )
    var weight_buffer = ctx.enqueue_create_buffer[dtype](
        weight_dynamic_shape.flattened_length()
    )
    var weight_device = LayoutTensor[dtype, weight_static_shape](
        weight_buffer, weight_runtime_layout
    )

    with weight_buffer.map_to_host() as weight_host:
        var weight_host_tensor = LayoutTensor[dtype, weight_static_shape](
            weight_host, weight_runtime_layout
        )
        random(weight_host_tensor)

    # Output tensor layout and buffer
    comptime output_static_shape = Layout.row_major(
        UNKNOWN_VALUE, combined_hidden_size
    )
    var output_dynamic_shape = IndexList[2](
        Int(total_seq_len), combined_hidden_size
    )
    var output_runtime_layout = RuntimeLayout[output_static_shape].row_major(
        output_dynamic_shape
    )
    var output_buffer = ctx.enqueue_create_buffer[dtype](
        output_dynamic_shape.flattened_length()
    )
    var output_device = LayoutTensor[dtype, output_static_shape](
        output_buffer, output_runtime_layout
    )

    # KV block tensor layout and buffer
    comptime kv_block_static_shape = Layout.row_major[6]()
    var kv_block_dynamic_shape = IndexList[6](
        num_blocks,
        2,
        num_layers,
        max_seq_length_cache,
        num_kv_heads,
        head_dim,
    )
    var kv_block_runtime_layout = RuntimeLayout[
        kv_block_static_shape
    ].row_major(kv_block_dynamic_shape)
    var kv_block_buffer = ctx.enqueue_create_buffer[dtype](
        kv_block_dynamic_shape.flattened_length()
    )
    var kv_block_device = LayoutTensor[dtype, kv_block_static_shape](
        kv_block_buffer, kv_block_runtime_layout
    )

    # Lookup table tensor layout and buffer
    comptime lookup_table_static_shape = Layout(UNKNOWN_VALUE)
    var lookup_table_dynamic_shape = IndexList[1](batch_size)
    var lookup_table_runtime_layout = RuntimeLayout[
        lookup_table_static_shape
    ].row_major(lookup_table_dynamic_shape)
    var lookup_table_buffer = ctx.enqueue_create_buffer[DType.uint32](
        lookup_table_dynamic_shape.flattened_length()
    )
    var lookup_table_device = LayoutTensor[
        DType.uint32, lookup_table_static_shape
    ](lookup_table_buffer, lookup_table_runtime_layout)

    # hacky way to select random blocks.
    with lookup_table_buffer.map_to_host() as lookup_table_host:
        var block_idx_set = Set[Int]()
        var idx = 0
        while idx < batch_size:
            var randval = Int(random_ui64(0, num_blocks - 1))
            if randval in block_idx_set:
                continue

            block_idx_set.add(randval)
            lookup_table_host[idx] = UInt32(randval)
            idx += 1

    # Cache lengths tensor layout and buffer
    comptime cache_lengths_static_shape = Layout(UNKNOWN_VALUE)
    var cache_lengths_dynamic_shape = IndexList[1](batch_size)
    var cache_lengths_runtime_layout = RuntimeLayout[
        cache_lengths_static_shape
    ].row_major(cache_lengths_dynamic_shape)
    var cache_lengths_buffer = ctx.enqueue_create_buffer[DType.uint32](
        cache_lengths_dynamic_shape.flattened_length()
    )
    var cache_lengths_device = LayoutTensor[
        DType.uint32, cache_lengths_static_shape
    ](cache_lengths_buffer, cache_lengths_runtime_layout)

    # Initialize cache lengths on host
    with cache_lengths_buffer.map_to_host() as cache_lengths_host:
        for i in range(batch_size):
            cache_lengths_host[i] = 10

    var kv_collection_device = CollectionType(
        LayoutTensor[dtype, kv_block_static_shape, MutAnyOrigin](
            kv_block_device.ptr,
            RuntimeLayout[kv_block_static_shape](
                kv_block_runtime_layout.shape.value,
                kv_block_runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[DType.uint32, cache_lengths_static_shape, ImmutAnyOrigin](
            cache_lengths_device.ptr,
            RuntimeLayout[cache_lengths_static_shape](
                cache_lengths_runtime_layout.shape.value,
                cache_lengths_runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[DType.uint32, lookup_table_static_shape, ImmutAnyOrigin](
            lookup_table_device.ptr,
            RuntimeLayout[lookup_table_static_shape](
                lookup_table_runtime_layout.shape.value,
                lookup_table_runtime_layout.stride.value,
            ),
        ),
        max_prompt_length,
        max_context_length,
    )

    var k_cache_device = kv_collection_device.get_key_cache(layer_idx)
    var v_cache_device = kv_collection_device.get_value_cache(layer_idx)

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
            _fused_qkv_matmul_kv_cache_ragged_impl[target="gpu"](
                hidden_state_device,
                prefix_sums_device_tensor,
                weight_device,
                k_cache_device,
                v_cache_device,
                output_device,
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
        [
            ThroughputMeasure(
                BenchMetric.flops,
                # Flop: 2*M*N*K. Use A and C shapes since they're not transposed.
                2 * Int(total_seq_len) * hidden_size * combined_hidden_size,
            )
        ],
    )


def main():
    comptime dtype = env_get_dtype["dtype", DType.bfloat16]()
    comptime head_dim = env_get_int["head_dim", 128]()
    comptime num_q_heads = env_get_int["num_q_heads", 128]()
    comptime num_kv_heads = env_get_int["num_kv_heads", 128]()

    var batch_size = arg_parse("batch_size", 1)
    var use_random_lengths = arg_parse("use_random_lengths", False)
    var seq_len = arg_parse("seq_len", 1)

    seed(0)

    var m = Bench()
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

    m.dump_report()
