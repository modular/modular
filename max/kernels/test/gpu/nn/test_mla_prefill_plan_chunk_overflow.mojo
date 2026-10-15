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
"""`mla_prefill_plan_kernel`'s chunk-overflow backstop.

`mla_prefill_plan_kernel` packs a batch's prefill work into ``MAX_CHUNKS``
buffer chunks, where ``MAX_CHUNKS`` is fixed by ``buffer_lengths``'s static
shape at graph-compile time. A sequence whose starting position lands in a
chunk at or beyond ``MAX_CHUNKS`` has no valid slot: the kernel's
``chunk_idx < start_chunk`` loop would silently mark every available chunk
as already consumed for that row, producing wrong gather offsets instead of
a visible failure.

The batch scheduler is expected to cap a batch's combined context so this
never happens in practice. This test drives the kernel-level backstop
directly and independently of that scheduler, by calling
``generic_flare_mla_prefill_ragged_paged_plan`` with a batch deliberately
sized to overflow a single-chunk buffer, and checks that the kernel aborts
loudly (via ``debug_assert``) instead of returning silently wrong offsets.
"""

from std.math import ceildiv
from std.memory import alloc
from std.utils.index import IndexList

from max.gpu.host import DeviceContext

from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Idx, TileTensor, row_major
from nn.kv_cache_ragged import generic_flare_mla_prefill_ragged_paged_plan

from _paged_prefill_test_utils import (
    CACHE_DEPTH,
    KV_NUM_HEADS,
    NUM_LAYERS,
    fill_paged_blocks_uniform,
    fill_uniform_lookup_table,
    lut_max_pages_per_batch,
    paged_block_elems,
)

comptime PAGE_SIZE = 128


def test_chunk_overflow(ctx: DeviceContext) raises:
    comptime batch_size = 2
    comptime MAX_CHUNKS = 1
    var seq_len = 9
    var cache_length = 500
    var num_keys = cache_length + seq_len  # 509, same for every sequence

    # Each sequence's page-aligned length is align_up(509, 128) = 512. A
    # buffer sized to a single page (PAGE_SIZE) gives the lone MAX_CHUNKS=1
    # slot room for only the first sequence: the second sequence's start
    # position (512) already lands in chunk 4, past the single available
    # slot, which is exactly the condition the backstop must catch.
    var buffer_token_size = UInt32(PAGE_SIZE)

    var num_pages = ceildiv(num_keys, PAGE_SIZE)
    var total_pages = batch_size * num_pages
    var max_pages_per_batch = lut_max_pages_per_batch(num_keys, PAGE_SIZE)
    var lut_size = batch_size * max_pages_per_batch
    var block_elems = paged_block_elems(total_pages, PAGE_SIZE, CACHE_DEPTH)

    var blocks_host = alloc[Scalar[.bfloat16]](block_elems)
    var cache_lengths_host = alloc[UInt32](batch_size)
    var lookup_table_host = alloc[UInt32](lut_size)

    fill_paged_blocks_uniform[.bfloat16](
        blocks_host, batch_size, num_keys, PAGE_SIZE
    )
    for b in range(batch_size):
        cache_lengths_host[b] = UInt32(cache_length)
    fill_uniform_lookup_table(
        lookup_table_host, batch_size, num_keys, PAGE_SIZE, max_pages_per_batch
    )

    var blocks_device = ctx.enqueue_create_buffer[.bfloat16](block_elems)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(blocks_device, blocks_host)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)
    ctx.synchronize()

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=CACHE_DEPTH, is_mla=True
    )
    var block_shape = IndexList[6](
        total_pages,
        1,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var blocks_tt = TileTensor(
        blocks_device,
        row_major(
            Int64(block_shape[0]),
            Idx[1],
            Int64(block_shape[2]),
            Idx[PAGE_SIZE],
            Idx[kv_params.num_heads],
            Idx[kv_params.head_size],
        ),
    )
    var cache_lengths_tt = TileTensor(
        cache_lengths_device, row_major(Int64(batch_size))
    )
    var lookup_table_tt = TileTensor(
        lookup_table_device,
        row_major(Int64(batch_size), Int64(max_pages_per_batch)),
    )
    comptime Collection = PagedKVCacheCollection[
        .bfloat16,
        kv_params,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var kv_collection = Collection(
        rebind[Collection.blocks_tt_type](blocks_tt.as_unsafe_any_origin()),
        cache_lengths_tt.as_imm().as_unsafe_any_origin(),
        lookup_table_tt.as_imm().as_unsafe_any_origin(),
        UInt32(seq_len),
        UInt32(num_keys),
    )

    var input_row_offsets_host = alloc[UInt32](batch_size + 1)
    for b in range(batch_size + 1):
        input_row_offsets_host[b] = UInt32(b * seq_len)
    var input_row_offsets_device = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )
    ctx.enqueue_copy(input_row_offsets_device, input_row_offsets_host)

    var buffer_row_offsets_device = ctx.enqueue_create_buffer[.uint32](
        MAX_CHUNKS * (batch_size + 1)
    )
    var cache_offsets_device = ctx.enqueue_create_buffer[.uint32](
        MAX_CHUNKS * batch_size
    )
    var buffer_lengths_device = ctx.enqueue_create_buffer[.int32](MAX_CHUNKS)
    ctx.synchronize()

    var input_row_offsets_tt = TileTensor(
        input_row_offsets_device, row_major(Int64(batch_size + 1))
    )
    var buffer_row_offsets_tt = TileTensor(
        buffer_row_offsets_device,
        row_major(Int64(MAX_CHUNKS), Int64(batch_size + 1)),
    )
    var cache_offsets_tt = TileTensor(
        cache_offsets_device,
        row_major(Int64(MAX_CHUNKS), Int64(batch_size)),
    )
    var buffer_lengths_tt = TileTensor(
        buffer_lengths_device, row_major(Idx[MAX_CHUNKS])
    )

    generic_flare_mla_prefill_ragged_paged_plan[target="gpu"](
        input_row_offsets_tt,
        kv_collection,
        UInt32(0),  # layer_idx
        buffer_token_size,
        buffer_row_offsets_tt,
        cache_offsets_tt,
        buffer_lengths_tt,
        ctx,
    )
    # Give the kernel's debug_assert an opportunity to fire before the test
    # binary exits.
    ctx.synchronize()

    blocks_host.free()
    cache_lengths_host.free()
    lookup_table_host.free()
    input_row_offsets_host.free()

    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = input_row_offsets_device
    _ = buffer_row_offsets_device
    _ = cache_offsets_device
    _ = buffer_lengths_device


def main() raises:
    with DeviceContext() as ctx:
        # CHECK: {{.*}}start_chunk exceeds MAX_CHUNKS{{.*}}
        test_chunk_overflow(ctx)
