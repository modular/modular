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
from math import ceildiv, rsqrt
from random import random_ui64, seed
from sys import env_get_dtype, env_get_int

from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext
from internal_utils import arg_parse
from layout._fillers import random
from memory import LegacyUnsafePointer as UnsafePointer
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from nn.mha import flash_attention
from nn.mha_mask import CausalMask
from nn.mha_score_mod import IdentityScoreMod
from tensor import IOUnknown, ManagedTensorSlice
from tensor.managed_tensor_slice import StaticTensorSpec

from utils import IndexList


def flops(
    batch: Int, nheads: Int, seqlen_q: Int, seqlen_k: Int, headdim: Int
) -> Int:
    var avg_seqlen = (max(0, seqlen_k - seqlen_q) + seqlen_k) / 2
    return Int(batch * nheads * 2 * seqlen_q * avg_seqlen * (headdim + headdim))


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
    return String(
        "fused_qkv_ragged_flash_attention",
        "(",
        dtype,
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
        ", use_random_seq_lengths=",
        use_random_seq_lengths,
        ", cache_len=",
        cache_len,
        ", use_random_cache_lengths=",
        use_random_cache_lengths,
    )


def execute_kv_cache_ragged_flash_attention[
    dtype: DType,
    head_dim: Int,
    num_q_heads: Int,
    num_kv_heads: Int,
    page_size: Int,
](
    ctx: DeviceContext,
    mut m: Bench,
    batch_size: Int,
    seq_len: Int,
    use_random_seq_lengths: Bool,
    cache_len: Int,
    use_random_cache_lengths: Bool,
    run_benchmark: Bool,
):
    comptime num_layers = 1
    comptime layer_idx = 0
    var num_pages = batch_size * ceildiv(seq_len + cache_len, page_size) * 2
    comptime CollectionType = PagedKVCacheCollection[
        dtype,
        KVCacheStaticParams(
            num_heads=UInt(num_kv_heads), head_size=UInt(head_dim)
        ),
        page_size,
    ]

    debug_assert(
        batch_size < num_pages,
        "batch_size passed to unit test (",
        batch_size,
        ") is larger than configured num_pages (",
        num_pages,
        ")",
    )

    # Host allocations for row offsets and cache lengths
    var input_row_offsets_host_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(
        batch_size + 1
    )
    var cache_lengths_host_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(
        batch_size
    )
    var max_context_length = 0
    var max_seq_length: UInt32 = 0
    var total_seq_len: UInt32 = 0
    var valid_lengths = List[Int]()

    for i in range(batch_size):
        var curr_seq_length: UInt32
        if use_random_seq_lengths:
            curr_seq_length = random_ui64(1, seq_len).cast[DType.uint32]()
        else:
            curr_seq_length = seq_len
        valid_lengths.append(Int(curr_seq_length))

        var curr_cache_length: UInt32
        if use_random_cache_lengths:
            curr_cache_length = random_ui64(0, cache_len).cast[DType.uint32]()
        else:
            curr_cache_length = cache_len

        curr_context_length = Int(curr_cache_length) + Int(curr_seq_length)

        max_context_length = max(max_context_length, curr_context_length)
        max_seq_length = max(max_seq_length, curr_seq_length)

        input_row_offsets_host_ptr[i] = total_seq_len
        cache_lengths_host_ptr[i] = curr_cache_length
        total_seq_len += curr_seq_length

    input_row_offsets_host_ptr[batch_size] = total_seq_len

    # Device allocations and copies for row offsets
    var input_row_offsets_dev_buffer = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    ctx.enqueue_copy(input_row_offsets_dev_buffer, input_row_offsets_host_ptr)
    var input_row_offsets_device = NDBuffer[DType.uint32, 1](
        input_row_offsets_dev_buffer.unsafe_ptr(),
        batch_size + 1,
    )

    # Device allocation and copy for cache lengths
    var cache_lengths_dev_buffer = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )
    ctx.enqueue_copy(cache_lengths_dev_buffer, cache_lengths_host_ptr)
    var cache_lengths_device = NDBuffer[DType.uint32, 1](
        cache_lengths_dev_buffer.unsafe_ptr(),
        batch_size,
    )

    # Q tensor allocation
    comptime static_q_shape = DimList(Dim(), num_q_heads, head_dim)
    var q_size = Int(total_seq_len) * num_q_heads * head_dim
    var q_host_ptr = UnsafePointer[Scalar[dtype]].alloc(q_size)
    var q_host = NDBuffer[dtype, 3, _, static_q_shape](
        q_host_ptr,
        DimList(Int(total_seq_len), num_q_heads, head_dim),
    )
    random(
        LayoutTensor[
            dtype, Layout.row_major(UNKNOWN_VALUE, num_q_heads, head_dim)
        ](
            q_host_ptr,
            RuntimeLayout[
                Layout.row_major(UNKNOWN_VALUE, num_q_heads, head_dim)
            ].row_major(
                IndexList[3](Int(total_seq_len), num_q_heads, head_dim)
            ),
        )
    )
    var q_dev_buffer = ctx.enqueue_create_buffer[dtype](q_size)
    ctx.enqueue_copy(q_dev_buffer, q_host_ptr)
    var q_device = NDBuffer[dtype, 3, _, static_q_shape](
        q_dev_buffer.unsafe_ptr(),
        DimList(Int(total_seq_len), num_q_heads, head_dim),
    )

    # Output tensor allocation
    var output_size = Int(total_seq_len) * num_q_heads * head_dim
    var output_host_ptr = UnsafePointer[Scalar[dtype]].alloc(output_size)
    var output_dev_buffer = ctx.enqueue_create_buffer[dtype](output_size)
    var output_device = NDBuffer[dtype, 3, _, static_q_shape](
        output_dev_buffer.unsafe_ptr(),
        DimList(Int(total_seq_len), num_q_heads, head_dim),
    )
    comptime output_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, head_dim
    )
    var output_device_tensor = LayoutTensor[dtype, output_layout, MutAnyOrigin](
        output_dev_buffer.unsafe_ptr(),
        RuntimeLayout[output_layout].row_major(
            IndexList[3](Int(total_seq_len), num_q_heads, head_dim)
        ),
    )
    # Paged LUT allocation
    var paged_lut_cols = ceildiv(max_context_length, page_size)
    var paged_lut_size = batch_size * paged_lut_cols
    var paged_lut_host_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(
        paged_lut_size
    )
    var paged_lut_host = NDBuffer[DType.uint32, 2](
        paged_lut_host_ptr,
        IndexList[2](batch_size, paged_lut_cols),
    )
    paged_lut_set = Set[Int]()
    for bs in range(batch_size):
        curr_seq_len = Int(cache_lengths_host_ptr[bs]) + valid_lengths[bs]
        for block_idx in range(0, ceildiv(curr_seq_len, page_size)):
            var randval = Int(random_ui64(0, num_pages - 1))
            while randval in paged_lut_set:
                randval = Int(random_ui64(0, num_pages - 1))

            paged_lut_set.add(randval)
            paged_lut_host[bs, block_idx] = randval

    var paged_lut_dev_buffer = ctx.enqueue_create_buffer[DType.uint32](
        paged_lut_size
    )
    ctx.enqueue_copy(paged_lut_dev_buffer, paged_lut_host_ptr)
    var paged_lut_device = NDBuffer[DType.uint32, 2](
        paged_lut_dev_buffer.unsafe_ptr(),
        IndexList[2](batch_size, paged_lut_cols),
    )

    # KV block paged allocation
    var kv_block_size = (
        num_pages * 2 * num_layers * page_size * num_kv_heads * head_dim
    )
    var kv_block_paged_host_ptr = UnsafePointer[Scalar[dtype]].alloc(
        kv_block_size
    )
    var kv_block_paged_host = NDBuffer[dtype, 6](
        kv_block_paged_host_ptr,
        IndexList[6](
            num_pages,
            2,
            num_layers,
            page_size,
            num_kv_heads,
            head_dim,
        ),
    )
    random(
        LayoutTensor[dtype, Layout.row_major[6](), MutAnyOrigin](
            kv_block_paged_host_ptr,
            RuntimeLayout[Layout.row_major[6]()].row_major(
                IndexList[6](
                    num_pages, 2, num_layers, page_size, num_kv_heads, head_dim
                )
            ),
        )
    )
    var kv_block_paged_dev_buffer = ctx.enqueue_create_buffer[dtype](
        kv_block_size
    )
    ctx.enqueue_copy(kv_block_paged_dev_buffer, kv_block_paged_host_ptr)
    var kv_block_paged_device = NDBuffer[dtype, 6](
        kv_block_paged_dev_buffer.unsafe_ptr(),
        IndexList[6](
            num_pages,
            2,
            num_layers,
            page_size,
            num_kv_heads,
            head_dim,
        ),
    )

    # Create LayoutTensors for KV collection
    comptime kv_block_layout = Layout.row_major[6]()
    var kv_block_layout_tensor = LayoutTensor[
        dtype, kv_block_layout, MutAnyOrigin
    ](
        kv_block_paged_dev_buffer.unsafe_ptr(),
        RuntimeLayout[kv_block_layout].row_major(
            IndexList[6](
                num_pages, 2, num_layers, page_size, num_kv_heads, head_dim
            )
        ),
    )

    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_layout_tensor = LayoutTensor[
        DType.uint32, cache_lengths_layout, ImmutAnyOrigin
    ](
        cache_lengths_dev_buffer.unsafe_ptr(),
        RuntimeLayout[cache_lengths_layout].row_major(IndexList[1](batch_size)),
    )

    comptime paged_lut_layout = Layout.row_major[2]()
    var paged_lut_layout_tensor = LayoutTensor[
        DType.uint32, paged_lut_layout, ImmutAnyOrigin
    ](
        paged_lut_dev_buffer.unsafe_ptr(),
        RuntimeLayout[paged_lut_layout].row_major(
            IndexList[2](batch_size, paged_lut_cols)
        ),
    )

    kv_collection_device = CollectionType(
        kv_block_layout_tensor,
        cache_lengths_layout_tensor,
        paged_lut_layout_tensor,
        max_seq_length,
        max_context_length,
    )

    k_cache_device = kv_collection_device.get_key_cache(layer_idx)
    v_cache_device = kv_collection_device.get_value_cache(layer_idx)

    # Create LayoutTensors for flash_attention inputs
    comptime q_layout = Layout.row_major(UNKNOWN_VALUE, num_q_heads, head_dim)
    var q_device_layout_tensor = LayoutTensor[dtype, q_layout, ImmutAnyOrigin](
        q_dev_buffer.unsafe_ptr(),
        RuntimeLayout[q_layout].row_major(
            IndexList[3](Int(total_seq_len), num_q_heads, head_dim)
        ),
    )

    comptime row_offsets_layout = Layout(UNKNOWN_VALUE)
    var input_row_offsets_layout_tensor = LayoutTensor[
        DType.uint32, row_offsets_layout, ImmutAnyOrigin
    ](
        input_row_offsets_dev_buffer.unsafe_ptr(),
        RuntimeLayout[row_offsets_layout].row_major(
            IndexList[1](batch_size + 1)
        ),
    )

    if run_benchmark:

        @parameter
        @__copy_capture(
            q_device_layout_tensor,
            k_cache_device,
            v_cache_device,
            output_device_tensor,
            input_row_offsets_layout_tensor,
        )
        @always_inline
        fn bench_func(mut b: Bencher):
            @parameter
            @always_inline
            fn kernel_launch(ctx: DeviceContext) raises:
                flash_attention[ragged=True](
                    output_device_tensor.as_any_origin(),
                    q_device_layout_tensor,
                    k_cache_device,
                    v_cache_device,
                    CausalMask(),
                    IdentityScoreMod(),
                    input_row_offsets_layout_tensor,
                    rsqrt(Float32(head_dim)),
                    ctx,
                )

            b.iter_custom[kernel_launch](ctx)

        flop_count = flops(
            batch_size,
            num_q_heads,
            seq_len,
            cache_len + seq_len,
            head_dim,
        )
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
    else:
        # `False` is useful for profiling with NCU.
        # We don't want to run the benchmark, as this makes the profiling
        # take a very long time and bloats the prof full of extra runs that
        # we don't look at.
        flash_attention[ragged=True](
            output_device_tensor.as_any_origin(),
            q_device_layout_tensor,
            k_cache_device,
            v_cache_device,
            CausalMask(),
            IdentityScoreMod(),
            input_row_offsets_layout_tensor,
            rsqrt(Float32(head_dim)),
            ctx,
        )

    # Cleanup host pointers
    input_row_offsets_host_ptr.free()
    cache_lengths_host_ptr.free()
    q_host_ptr.free()
    output_host_ptr.free()
    paged_lut_host_ptr.free()
    kv_block_paged_host_ptr.free()

    # Consume device buffers
    _ = input_row_offsets_dev_buffer^
    _ = cache_lengths_dev_buffer^
    _ = q_dev_buffer^
    _ = output_dev_buffer^
    _ = paged_lut_dev_buffer^
    _ = kv_block_paged_dev_buffer^


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
    var run_benchmark = arg_parse("run_benchmark", True)

    seed(0)

    var m = Bench()
    try:
        with DeviceContext() as ctx:
            # benchmarking flash attention
            execute_kv_cache_ragged_flash_attention[
                dtype,
                head_dim,
                num_q_heads,
                num_kv_heads,
                512,
            ](
                ctx,
                m,
                batch_size,
                seq_len,
                use_random_seq_lengths,
                cache_len,
                use_random_cache_lengths,
                run_benchmark,
            )

    except e:
        print("CUDA_ERROR:", e)

    m.dump_report()
