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
"""KERN-3412: ``_k_cache_to_buffer`` must write zero, not gather, for rows
past the plan's real total, when its ``length`` argument (this launch's
row extent) is larger than that total -- as it will be once callers size
that extent to a capture-safe upper bound rather than exactly to a given
step's real span.

``test_mla_prefill_amd_plan_gather.mojo`` (this directory) already covers the
in-range page-alignment tail (rows between ``num_keys`` and
``align_up(num_keys, page_size)``, still within the sequence's own allocated
pages). This test covers the different, out-of-range case: rows the plan's
own total does not include at all, manufactured by calling the gather with a
``buffer_length`` larger than what the plan actually computed.

Why the paged cache's own design makes this deterministic
===========================================================

For those out-of-range rows, ``get_batch_from_row_offsets`` clamps to the
last batch and ``token_idx`` runs past that sequence's own cache span. The
paged lookup table's unassigned-slot columns hold a sentinel block index
(production convention: ``total_num_pages``, filled by
``paged_kv_cache/cache_manager.py``'s ``lut_table_np.fill(...)``, landing on
one extra page allocated past every real page precisely so this resolves
in-bounds rather than faulting). This test reproduces that exact convention
by hand: it allocates one extra "null" page beyond this sequence's real
pages, points every unused LUT column at it, and fills it with a distinctive
value no real latent data would produce.

Because the null page's content is under this test's control, the check
doesn't need to guess what "arbitrary" memory contains: it pre-fills the
gather's OUTPUT with a sentinel distinct from both zero and the null
page's value, then asserts every out-of-range row is exactly zero
afterward -- proof the guard wrote it, not that the row happened to start
at zero. Without the fix, ``_k_cache_to_buffer`` overwrites those rows
with the null page's content instead.

Shapes
======

Same as the sibling plan+gather test: ``page_size=128``, ``batch_size=1``,
``cache_length=500``, ``seq_len=9`` (``num_keys=509``), so
``buffer_length = align_up(509, 128) = 512`` and ``num_pages_per_batch=4``
out of ``max_pages_per_batch=8`` -- four spare LUT columns, one of which
this test points at the null page. The out-of-range span tested is one
full extra page (``[512, 640)``), which lands entirely on that one
sentinel LUT column.
"""

from std.math import align_up, ceildiv
from std.memory import alloc
from std.random import randn, seed
from std.sys import get_defined_int, has_amd_gpu_accelerator
from std.testing import assert_almost_equal
from std.utils.index import IndexList

from max.gpu.host import DeviceContext

from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Coord, Idx, TileTensor, row_major
from nn.kv_cache_ragged import (
    generic_flare_mla_decompress_k_cache_ragged_paged,
    generic_flare_mla_prefill_ragged_paged_plan,
)

from _paged_prefill_test_utils import (
    CACHE_DEPTH,
    KV_NUM_HEADS,
    NUM_LAYERS,
    ROPE_DEPTH,
    fill_paged_blocks_uniform,
    fill_uniform_lookup_table,
    page_stride,
    paged_block_elems,
    token_stride,
)

comptime LATENT_DIM = CACHE_DEPTH - ROPE_DEPTH  # 512: kv_lora_rank
comptime PAGE_SIZE = get_defined_int["page_size", 128]()
comptime OUT_DIM = 128  # one up-projection column block, matches the sibling
comptime NULL_PAGE_VALUE = 999.0
comptime PRE_FILL_SENTINEL = -1.0


def main() raises:
    with DeviceContext() as ctx:
        comptime if has_amd_gpu_accelerator():
            test_gather_zeros_out_of_range_rows(ctx)


def test_gather_zeros_out_of_range_rows(ctx: DeviceContext) raises:
    comptime batch_size = 1
    comptime MAX_CHUNKS = 1
    var cache_length = 500
    var seq_len = 9
    var num_keys = cache_length + seq_len  # 509, not page-aligned
    var num_pages = ceildiv(num_keys, PAGE_SIZE)  # 4 real pages
    var null_page = num_pages  # one extra page past every real page
    var total_pages = null_page + 1  # 5
    var max_pages_per_batch = align_up(num_pages, 8)  # 8: spare LUT columns

    print(
        "=== gather sentinel guard: cache_length=",
        cache_length,
        " seq_len=",
        seq_len,
        " num_keys=",
        num_keys,
        " ===",
    )
    seed(0)

    # ------------------------------------------------------------------
    # Step 1: paged latent+rope cache, with a real allocated "null" page
    # past this sequence's own pages -- mirrors the production
    # `total_num_pages + 1` convention (`paged_kv_cache/cache_manager.py`)
    # so the sentinel resolves to a page with known content instead of an
    # unwritten LUT slot.
    # ------------------------------------------------------------------
    var block_elems = paged_block_elems(total_pages, PAGE_SIZE, CACHE_DEPTH)
    var lut_size = batch_size * max_pages_per_batch

    var blocks_host = alloc[Scalar[.bfloat16]](block_elems)
    var cache_lengths_host = alloc[UInt32](batch_size)
    var lookup_table_host = alloc[UInt32](lut_size)

    # Real pages [0, num_pages): random content + zero-filled page-alignment
    # tail, exactly as the sibling plan+gather test uses. This helper only
    # touches those pages; the extra null page past them is filled next.
    fill_paged_blocks_uniform[.bfloat16](
        blocks_host, batch_size, num_keys, PAGE_SIZE
    )
    var pstride = page_stride(PAGE_SIZE, CACHE_DEPTH)
    var tstride = token_stride(CACHE_DEPTH)
    var null_base = null_page * pstride
    for z in range(PAGE_SIZE * tstride):
        blocks_host[null_base + z] = NULL_PAGE_VALUE

    cache_lengths_host[0] = UInt32(cache_length)
    # Every spare LUT column is repointed at the null page instead of the
    # helper's usual zero.
    fill_uniform_lookup_table(
        lookup_table_host, batch_size, num_keys, PAGE_SIZE, max_pages_per_batch
    )
    for p in range(num_pages, max_pages_per_batch):
        lookup_table_host[p] = UInt32(null_page)

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

    # ------------------------------------------------------------------
    # Step 2: plan. Gives the real total (`buffer_row_offsets`'s last
    # entry / `buffer_lengths[0]`) that the guard must key on.
    # ------------------------------------------------------------------
    var input_row_offsets_host = alloc[UInt32](batch_size + 1)
    input_row_offsets_host[0] = UInt32(0)
    input_row_offsets_host[1] = UInt32(seq_len)
    var input_row_offsets_device = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )
    ctx.enqueue_copy(input_row_offsets_device, input_row_offsets_host)

    var buffer_token_size = UInt32(align_up(num_keys, PAGE_SIZE))

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
    # The plan kernel unrolls MAX_CHUNKS from this static shape.
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
    ctx.synchronize()

    var buffer_lengths_host = alloc[Int32](MAX_CHUNKS)
    ctx.enqueue_copy(buffer_lengths_host, buffer_lengths_device)
    ctx.synchronize()

    var real_total = Int(buffer_lengths_host[0])
    var expect_real_total = align_up(num_keys, PAGE_SIZE)
    if real_total != expect_real_total:
        raise Error(
            "plan: expected buffer_lengths[0]="
            + String(expect_real_total)
            + ", got "
            + String(real_total)
        )

    # ------------------------------------------------------------------
    # Step 3: gather, with `buffer_length` inflated by one full page past
    # the plan's real total -- the scenario a capture-safe upper bound
    # creates. Every row in that extra page maps to the same sentinel LUT
    # column (`real_total` is already page-aligned), so this is a single,
    # deterministic out-of-range span.
    # ------------------------------------------------------------------
    var inflated_length = real_total + PAGE_SIZE
    print(
        "  real_total=",
        real_total,
        " inflated_length=",
        inflated_length,
    )

    var buffer_row_offsets_1d_tt = buffer_row_offsets_tt[0, :]
    var cache_offsets_1d_tt = cache_offsets_tt[0, :]

    var weight_host = alloc[Scalar[.bfloat16]](OUT_DIM * LATENT_DIM)
    randn[.bfloat16](weight_host, OUT_DIM * LATENT_DIM)
    var weight_device = ctx.enqueue_create_buffer[.bfloat16](
        OUT_DIM * LATENT_DIM
    )
    ctx.enqueue_copy(weight_device, weight_host)

    var k_latent_device = ctx.enqueue_create_buffer[.bfloat16](
        inflated_length * LATENT_DIM
    )
    var k_buffer_device = ctx.enqueue_create_buffer[.bfloat16](
        inflated_length * OUT_DIM
    )
    # Sentinel distinct from zero and the null page's value: proves the
    # zero asserted below came from an active write, not luck.
    var sentinel_host = alloc[Scalar[.bfloat16]](inflated_length * LATENT_DIM)
    for i in range(inflated_length * LATENT_DIM):
        sentinel_host[i] = PRE_FILL_SENTINEL
    ctx.enqueue_copy(k_latent_device, sentinel_host)
    ctx.synchronize()

    var weight_tt = TileTensor(
        weight_device, row_major(Int64(OUT_DIM), Int64(LATENT_DIM))
    )
    var k_latent_tt = TileTensor(
        k_latent_device,
        row_major(Int64(inflated_length), Idx[LATENT_DIM]),
    )
    var k_buffer_tt = TileTensor(
        k_buffer_device, row_major(Int64(inflated_length), Idx[OUT_DIM])
    )

    generic_flare_mla_decompress_k_cache_ragged_paged[
        target="gpu", dtype=DType.bfloat16
    ](
        buffer_row_offsets_1d_tt,
        cache_offsets_1d_tt,
        Int32(inflated_length),
        weight_tt,
        kv_collection,
        UInt32(0),  # layer_idx
        k_latent_tt,
        k_buffer_tt,
        ctx,
    )
    ctx.synchronize()

    var k_latent_host = alloc[Scalar[.bfloat16]](inflated_length * LATENT_DIM)
    ctx.enqueue_copy(k_latent_host, k_latent_device)
    ctx.synchronize()

    # ------------------------------------------------------------------
    # Step 4: checks.
    # ------------------------------------------------------------------
    # (a) In-range rows still gather real data -- the guard must not
    # touch anything it didn't touch before.
    comptime atol: Float64 = 1e-2
    comptime rtol: Float64 = 1e-2
    for r in range(real_total):
        var page_idx = r // PAGE_SIZE
        var tok_in_page = r % PAGE_SIZE
        var src_base = page_idx * pstride + tok_in_page * tstride
        for d in range(LATENT_DIM):
            var actual = k_latent_host[r * LATENT_DIM + d].cast[.float64]()
            var expect = blocks_host[src_base + d].cast[.float64]()
            assert_almost_equal(actual, expect, atol=atol, rtol=rtol)

    # (b) Out-of-range rows. A leftover `PRE_FILL_SENTINEL` means the row
    # was never written (no guard); a value near `NULL_PAGE_VALUE` means
    # it was gathered from the null page (guard missing the check).
    var n_nonzero = 0
    for r in range(real_total, inflated_length):
        for d in range(LATENT_DIM):
            var actual = k_latent_host[r * LATENT_DIM + d]
            if actual.cast[.float64]() != 0.0:
                n_nonzero += 1
    if n_nonzero > 0:
        raise Error(
            String(n_nonzero)
            + " element(s) in the out-of-range span [real_total="
            + String(real_total)
            + ", inflated_length="
            + String(inflated_length)
            + ") were not zero -- `_k_cache_to_buffer` did not write zero"
            + " to every row past the plan's real total."
        )

    print("  RESULT: PASS")

    blocks_host.free()
    cache_lengths_host.free()
    lookup_table_host.free()
    input_row_offsets_host.free()
    buffer_lengths_host.free()
    weight_host.free()
    sentinel_host.free()
    k_latent_host.free()

    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = input_row_offsets_device
    _ = buffer_row_offsets_device
    _ = cache_offsets_device
    _ = buffer_lengths_device
    _ = weight_device
    _ = k_latent_device
    _ = k_buffer_device
