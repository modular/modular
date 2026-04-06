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

from std.math import ceildiv
from std.random import seed
from std.sys import get_defined_dtype, get_defined_int
from std.sys.info import size_of

from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from std.gpu.host import DeviceContext, get_gpu_target
from internal_utils import arg_parse
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from layout._fillers import random
from nn.rope import _rope_k_cache_ragged

from std.utils import IndexList


def _get_run_name[
    dtype: DType, num_kv_heads: Int, head_dim: Int, page_size: Int
](batch_size: Int, seq_len: Int, cache_len: Int) -> String:
    return String(
        "k_cache_rope(",
        dtype,
        ") : num_kv_heads=",
        num_kv_heads,
        ", head_dim=",
        head_dim,
        ", page_size=",
        page_size,
        " : batch_size=",
        batch_size,
        ", seq_len=",
        seq_len,
        ", cache_len=",
        cache_len,
    )


def execute_kv_cache_ragged_k_rope[
    dtype: DType, head_dim: Int, num_kv_heads: Int, page_size: Int
](
    ctx: DeviceContext,
    mut m: Bench,
    batch_size: Int,
    seq_len: Int,
    cache_len: Int,
) raises:
    comptime num_layers = 1
    comptime layer_idx = 0
    comptime kv_params = KVCacheStaticParams(
        num_heads=UInt(num_kv_heads), head_size=UInt(head_dim)
    )
    comptime CollectionType = PagedKVCacheCollection[dtype, kv_params, page_size]

    var total_seq_len = UInt32(batch_size * seq_len)
    var max_context_length = seq_len + cache_len
    var paged_lut_cols = ceildiv(max_context_length, page_size)
    var num_pages = batch_size * paged_lut_cols

    var input_row_offsets_host_ptr = alloc[Scalar[DType.uint32]](batch_size + 1)
    var cache_lengths_host_ptr = alloc[Scalar[DType.uint32]](batch_size)

    for i in range(batch_size):
        input_row_offsets_host_ptr[i] = UInt32(i * seq_len)
        cache_lengths_host_ptr[i] = UInt32(cache_len)
    input_row_offsets_host_ptr[batch_size] = total_seq_len

    var input_row_offsets_dev_buffer = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    ctx.enqueue_copy(input_row_offsets_dev_buffer, input_row_offsets_host_ptr)

    var cache_lengths_dev_buffer = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )
    ctx.enqueue_copy(cache_lengths_dev_buffer, cache_lengths_host_ptr)

    var paged_lut_size = batch_size * paged_lut_cols
    var paged_lut_host_ptr = alloc[Scalar[DType.uint32]](paged_lut_size)
    comptime paged_lut_layout = Layout.row_major[2]()
    var paged_lut_host = LayoutTensor[
        DType.uint32, paged_lut_layout, MutAnyOrigin
    ](
        paged_lut_host_ptr,
        RuntimeLayout[paged_lut_layout].row_major(
            IndexList[2](batch_size, paged_lut_cols)
        ),
    )

    # Assign each request a deterministic, non-overlapping page range.
    for bs in range(batch_size):
        for page_idx in range(paged_lut_cols):
            paged_lut_host[bs, page_idx] = UInt32(bs * paged_lut_cols + page_idx)

    var paged_lut_dev_buffer = ctx.enqueue_create_buffer[DType.uint32](
        paged_lut_size
    )
    ctx.enqueue_copy(paged_lut_dev_buffer, paged_lut_host_ptr)

    var kv_block_shape = IndexList[6](
        num_pages,
        2,
        num_layers,
        page_size,
        num_kv_heads,
        head_dim,
    )
    var kv_block_size = kv_block_shape.flattened_length()
    var kv_block_host_ptr = alloc[Scalar[dtype]](kv_block_size)
    random(
        LayoutTensor[dtype, Layout.row_major[6](), MutAnyOrigin](
            kv_block_host_ptr,
            RuntimeLayout[Layout.row_major[6]()].row_major(kv_block_shape),
        )
    )
    var kv_block_dev_buffer = ctx.enqueue_create_buffer[dtype](kv_block_size)
    ctx.enqueue_copy(kv_block_dev_buffer, kv_block_host_ptr)

    comptime kv_block_layout = Layout.row_major[6]()
    var kv_block_layout_tensor = LayoutTensor[
        dtype, kv_block_layout, MutAnyOrigin
    ](
        kv_block_dev_buffer.unsafe_ptr(),
        RuntimeLayout[kv_block_layout].row_major(kv_block_shape),
    )

    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_layout_tensor = LayoutTensor[
        DType.uint32, cache_lengths_layout, ImmutAnyOrigin
    ](
        cache_lengths_dev_buffer.unsafe_ptr(),
        RuntimeLayout[cache_lengths_layout].row_major(IndexList[1](batch_size)),
    )

    var paged_lut_layout_tensor = LayoutTensor[
        DType.uint32, paged_lut_layout, ImmutAnyOrigin
    ](
        paged_lut_dev_buffer.unsafe_ptr(),
        RuntimeLayout[paged_lut_layout].row_major(
            IndexList[2](batch_size, paged_lut_cols)
        ),
    )

    var kv_collection = CollectionType(
        kv_block_layout_tensor,
        cache_lengths_layout_tensor,
        paged_lut_layout_tensor,
        UInt32(seq_len),
        UInt32(max_context_length),
    )

    var input_row_offsets_tensor = TileTensor(
        input_row_offsets_dev_buffer.unsafe_ptr(),
        row_major(Idx(batch_size + 1)),
    )
    var freqs_cis_layout = row_major(Idx(max_context_length), Idx[head_dim]())
    var freqs_cis_device = ctx.enqueue_create_buffer[dtype](
        max_context_length * head_dim
    )
    with freqs_cis_device.map_to_host() as freqs_host:
        random(TileTensor(freqs_host, freqs_cis_layout))
    var freqs_cis_tensor = TileTensor(
        freqs_cis_device.unsafe_ptr(), freqs_cis_layout
    )

    var num_bytes = (
        3 * Int(total_seq_len) * num_kv_heads * head_dim * size_of[dtype]()
    )

    @parameter
    @__copy_capture(kv_collection, input_row_offsets_tensor, freqs_cis_tensor)
    @always_inline
    def bench_func(mut b: Bencher):
        @parameter
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises:
            _rope_k_cache_ragged[target="gpu", interleaved=False](
                Int(total_seq_len),
                input_row_offsets_tensor,
                kv_collection,
                freqs_cis_tensor,
                UInt32(layer_idx),
                ctx,
            )

        b.iter_custom[kernel_launch](ctx)

    m.bench_function[bench_func](
        BenchId(
            _get_run_name[dtype, num_kv_heads, head_dim, page_size](
                batch_size, seq_len, cache_len
            )
        ),
        [ThroughputMeasure(BenchMetric.bytes, num_bytes)],
    )
    ctx.synchronize()

    input_row_offsets_host_ptr.free()
    cache_lengths_host_ptr.free()
    paged_lut_host_ptr.free()
    kv_block_host_ptr.free()

    _ = input_row_offsets_dev_buffer^
    _ = cache_lengths_dev_buffer^
    _ = paged_lut_dev_buffer^
    _ = kv_block_dev_buffer^
    _ = freqs_cis_device^


def main() raises:
    comptime dtype = get_defined_dtype["dtype", DType.bfloat16]()
    comptime head_dim = get_defined_int["head_dim", 128]()
    comptime num_kv_heads = get_defined_int["num_kv_heads", 16]()

    var batch_size = arg_parse("batch_size", 1)
    var seq_len = arg_parse("seq_len", 1)
    var cache_len = arg_parse("cache_len", 0)

    seed(0)

    var m = Bench()
    with DeviceContext() as ctx:
        execute_kv_cache_ragged_k_rope[
            dtype,
            head_dim,
            num_kv_heads,
            512,
        ](ctx, m, batch_size, seq_len, cache_len)

    m.dump_report()
