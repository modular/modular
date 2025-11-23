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
from internal_utils import HostNDBuffer, arg_parse, random
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
)
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
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
    var prefix_sums_host = HostNDBuffer[DType.uint32, 1](
        DimList(batch_size + 1),
    )

    for i in range(batch_size):
        var length: UInt32
        if use_random_lengths:
            length = random_ui64(1, seq_len).cast[DType.uint32]()
        else:
            length = seq_len

        prefix_sums_host.tensor[i] = length
        total_seq_len += length
        max_context_length = max(max_context_length, Int(length + cache_size))
        max_prompt_length = max(max_prompt_length, Int(length))
    prefix_sums_host.tensor[batch_size] = total_seq_len
    var prefix_sums_device_buffer = prefix_sums_host.copy_to_device(ctx)
    var prefix_sums_device = prefix_sums_device_buffer.to_layout_tensor()
    var hidden_state_host = HostNDBuffer[dtype, 2, DimList(Dim(), hidden_size)](
        (Int(total_seq_len), hidden_size),
    )
    random(hidden_state_host.tensor)
    var hidden_state_device_buffer = hidden_state_host.copy_to_device(ctx)
    var hidden_state_device = hidden_state_device_buffer.to_layout_tensor()

    var weight_host = HostNDBuffer[
        dtype, 2, DimList(hidden_size, combined_hidden_size)
    ]((hidden_size, combined_hidden_size))
    random(weight_host.tensor)
    var weight_device_buffer = weight_host.copy_to_device(ctx)
    var weight_device = weight_device_buffer.to_layout_tensor()

    var output_host = HostNDBuffer[dtype, 2, DimList(Dim(), hidden_size)](
        (Int(total_seq_len), combined_hidden_size),
    )
    random(output_host.tensor)
    var output_devce_buffer = output_host.copy_to_device(ctx)
    var output_device = output_devce_buffer.to_layout_tensor()

    var kv_block_host = HostNDBuffer[dtype, 6](
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

    # initialize our KVCache
    var cache_lengths_host = HostNDBuffer[DType.uint32, 1](DimList(batch_size))
    for i in range(batch_size):
        cache_lengths_host.tensor[i] = 10

    var cache_lengths_device = cache_lengths_host.copy_to_device(ctx)

    var kv_collection_device = CollectionType(
        LayoutTensor[
            kv_block_device.dtype, Layout.row_major[6](), MutAnyOrigin
        ](
            kv_block_device.to_layout_tensor().ptr,
            RuntimeLayout[Layout.row_major[6]()](
                kv_block_device.to_layout_tensor().runtime_layout.shape.value,
                kv_block_device.to_layout_tensor().runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[
            cache_lengths_device.dtype, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
        ](
            cache_lengths_device.to_layout_tensor().ptr,
            RuntimeLayout[Layout(UNKNOWN_VALUE)](
                cache_lengths_device.to_layout_tensor().runtime_layout.shape.value,
                cache_lengths_device.to_layout_tensor().runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[
            lookup_table_device.dtype, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
        ](
            lookup_table_device.to_layout_tensor().ptr,
            RuntimeLayout[Layout(UNKNOWN_VALUE)](
                lookup_table_device.to_layout_tensor().runtime_layout.shape.value,
                lookup_table_device.to_layout_tensor().runtime_layout.stride.value,
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
                prefix_sums_device,
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
