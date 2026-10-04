# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from std.collections import Set
from std.random import random_ui64, seed
from std.sys import get_defined_dtype, get_defined_int

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceContext
from internal_utils import arg_parse
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
)
from layout import (
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from layout._fillers import random
from nn.kv_cache_ragged import _fused_qkv_matmul_kv_cache_ragged_impl
from std.utils import IndexList


def _get_run_name[
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
) raises:
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
    var prefix_sums_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var prefix_sums_device_tensor = TileTensor(
        prefix_sums_device, row_major(batch_size + 1)
    )

    with prefix_sums_device.map_to_host() as prefix_sums_host:
        for i in range(batch_size):
            var length: UInt32
            if use_random_lengths:
                length = random_ui64(1, UInt64(seq_len)).cast[.uint32]()
            else:
                length = UInt32(seq_len)

            prefix_sums_host[i] = length
            total_seq_len += length
            max_context_length = max(
                max_context_length, Int(length + cache_size)
            )
            max_prompt_length = max(max_prompt_length, Int(length))
        prefix_sums_host[batch_size] = total_seq_len
    # Hidden state tensor layout and buffer
    var hidden_state_buffer = ctx.enqueue_create_buffer[dtype](
        Int(total_seq_len) * hidden_size
    )
    var hidden_state_device = TileTensor(
        hidden_state_buffer,
        row_major((total_seq_len, Idx[hidden_size])),
    )

    with hidden_state_buffer.map_to_host() as hidden_state_host:
        var hidden_state_host_tensor = TileTensor(
            hidden_state_host,
            row_major((total_seq_len, Idx[hidden_size])),
        )
        random(hidden_state_host_tensor)

    # Weight tensor layout and buffer
    var weight_buffer = ctx.enqueue_create_buffer[dtype](
        hidden_size * combined_hidden_size
    )
    var weight_device = TileTensor(
        weight_buffer,
        row_major((Idx[hidden_size], Idx[combined_hidden_size])),
    )

    with weight_buffer.map_to_host() as weight_host:
        var weight_host_tensor = TileTensor(
            weight_host,
            row_major((Idx[hidden_size], Idx[combined_hidden_size])),
        )
        random(weight_host_tensor)

    # Output tensor layout and buffer
    var output_buffer = ctx.enqueue_create_buffer[dtype](
        Int(total_seq_len) * combined_hidden_size
    )
    var output_device = TileTensor(
        output_buffer,
        row_major((total_seq_len, Idx[combined_hidden_size])),
    )

    # KV block tensor layout and buffer
    var kv_block_dynamic_shape = IndexList[6](
        num_blocks,
        2,
        num_layers,
        max_seq_length_cache,
        num_kv_heads,
        head_dim,
    )
    var kv_block_buffer = ctx.enqueue_create_buffer[dtype](
        kv_block_dynamic_shape.flattened_length()
    )
    comptime Collection = ContinuousBatchingKVCacheCollection[
        dtype,
        KVCacheStaticParams(num_heads=num_kv_heads, head_size=head_dim),
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
    ]
    comptime BlocksLayout = Collection.blocks_tt_layout
    var blocks_shape = Coord[*BlocksLayout.shape_types]()
    blocks_shape[0] = Int64(num_blocks)
    blocks_shape[1] = Int64(2)
    blocks_shape[2] = Int64(num_layers)
    blocks_shape[3] = Int64(max_seq_length_cache)
    var blocks_strides = Coord[*BlocksLayout.stride_types]()
    blocks_strides[0] = Int64(
        2 * num_layers * max_seq_length_cache * num_kv_heads * head_dim
    )
    blocks_strides[1] = Int64(
        num_layers * max_seq_length_cache * num_kv_heads * head_dim
    )
    blocks_strides[2] = Int64(max_seq_length_cache * num_kv_heads * head_dim)
    var kv_block_device = TileTensor(
        kv_block_buffer, row_major(len(kv_block_buffer))
    ).reshape(BlocksLayout(blocks_shape, blocks_strides))

    var lookup_table_buffer = ctx.enqueue_create_buffer[.uint32](batch_size)
    var lookup_table_device = TileTensor(
        lookup_table_buffer, row_major(len(lookup_table_buffer))
    ).reshape(Coord(Int64(batch_size)))

    # Sample distinct physical blocks so sequences do not alias KV storage.
    with lookup_table_buffer.map_to_host() as lookup_table_host:
        var block_idx_set = Set[Int]()
        var idx = 0
        while idx < batch_size:
            var randval = Int(random_ui64(0, UInt64(num_blocks - 1)))
            if randval in block_idx_set:
                continue

            block_idx_set.add(randval)
            lookup_table_host[idx] = UInt32(randval)
            idx += 1

    var cache_lengths_buffer = ctx.enqueue_create_buffer[.uint32](batch_size)
    var cache_lengths_device = TileTensor(
        cache_lengths_buffer, row_major(len(cache_lengths_buffer))
    ).reshape(Coord(Int64(batch_size)))

    # Initialize cache lengths on host
    with cache_lengths_buffer.map_to_host() as cache_lengths_host:
        for i in range(batch_size):
            cache_lengths_host[i] = 10

    # K and V occupy disjoint regions per block; the launch writes both.
    var kv_collection_device = Collection(
        kv_block_device.as_unsafe_any_origin(),
        cache_lengths_device.as_imm().as_unsafe_any_origin(),
        lookup_table_device.as_imm().as_unsafe_any_origin(),
        UInt32(max_prompt_length),
        UInt32(max_context_length),
    )

    var k_cache_device = kv_collection_device.get_key_cache(layer_idx)
    var v_cache_device = kv_collection_device.get_value_cache(layer_idx)

    @inline(.always)
    def bench_func(
        mut b: Bencher,
    ) raises {
        var hidden_state_device,
        var prefix_sums_device,
        var k_cache_device,
        var v_cache_device,
        var output_device,
        imm,
    }:
        @inline(.always)
        def kernel_launch(ctx: DeviceContext) raises {imm}:
            _fused_qkv_matmul_kv_cache_ragged_impl[target="gpu"](
                hidden_state_device.as_imm().as_unsafe_any_origin(),
                prefix_sums_device_tensor.as_imm(),
                weight_device.as_imm().as_unsafe_any_origin(),
                k_cache_device,
                v_cache_device,
                output_device,
                ctx,
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    m.bench_function(
        bench_func,
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


def main() raises:
    comptime dtype = get_defined_dtype["dtype", .bfloat16]()
    comptime head_dim = get_defined_int["head_dim", 128]()
    comptime num_q_heads = get_defined_int["num_q_heads", 128]()
    comptime num_kv_heads = get_defined_int["num_kv_heads", 128]()

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
