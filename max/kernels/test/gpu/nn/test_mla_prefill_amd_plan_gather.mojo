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
"""AMD/gfx950 MLA prefill plan + gather/decompress stages, end to end.

``test_mla_prefill_amd_cached_prefix.mojo`` (this directory) drives only the
THIRD of production's three MLA prefill stages:

  1. plan (``generic_flare_mla_prefill_ragged_paged_plan`` ->
     ``mla_prefill_plan``): computes ``buffer_row_offsets`` /
     ``cache_offsets`` / ``buffer_lengths`` from ``input_row_offsets`` and
     the paged cache's own per-batch ``cache_length``.
  2. gather + decompress (``generic_flare_mla_decompress_k_cache_ragged_paged``
     -> ``_k_cache_to_buffer`` + a matmul): gathers the latent K/V rows named by
     stage 1 into a contiguous buffer, then up-projects them via a weight
     matrix into the decompressed K-nope/V buffer the attention kernel reads.
  3. attention (``generic_flare_mla_prefill_kv_cache_ragged`` ->
     ``flare_mla_prefill[rank=3]``): the only stage the other file's harness
     (and every existing SM100 paged-prefill test) reaches. Its K-nope/V
     inputs there are directly `randn`-filled flat buffers -- never gathered,
     never paged.

So a defect in stage 1 or 2 is invisible to that file, at any cache_length,
page size, or LUT layout, since none of ``buffer_row_offsets``,
``cache_offsets``, ``_k_cache_to_buffer``, or ``mla_prefill_plan`` are ever
exercised there. This test drives stages 1 and 2 directly and checks the
decompressed output against a reference computed from the same latent data
independently of the kernel path, closing that gap. It does not re-drive
stage 3 (attention) -- that is already covered, including the cached-prefix
and scattered-LUT geometries, by the sibling file.

Why stage 3 isn't needed here
==============================

Stages 1 and 2 have no dependence on Q, the attention mask, or per-head
splitting beyond a fixed output width: `_k_cache_to_buffer`'s gather is
per-token and per-column, and the decompression is a single matmul
(``k_buffer = k_latent_buffer @ weight^T``). A reference that reproduces
this arithmetic directly on the same latent content the paged cache holds
is a complete correctness check for these two stages without needing a
causal-attention reference at all. `out_dim=128` here (one up-projection
column block, not all 12 of K3's real heads) since the per-head replication
is the same per-token operation repeated -- it doesn't exercise anything
stage 1 or 2 doesn't already have to get right for a single block.

Shapes
======

``page_size=128``, ``batch_size=1``, ``seq_len=9`` (K3's DSpark verify
window). Two ``cache_length`` cases, both deliberately NOT a multiple of
``page_size``:

  - ``cache_length=500`` -> ``num_keys=509`` (``align_up(509,128)=512``,
    a 3-row tail).
  - ``cache_length=900`` -> ``num_keys=909`` (``align_up(909,128)=1024``,
    a 115-row tail) -- a larger page count, since the over-read below
    scales with it.

That tail is where `mla_prefill_plan`'s page-alignment makes stage 2
gather past the true valid count.
``fill_paged_blocks_uniform`` zero-fills the cache past ``num_keys``, so
this test asserts the decompressed buffer's tail rows are exactly zero
rather than assuming the over-read is harmless.
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
    lut_max_pages_per_batch,
    page_stride,
    paged_block_elems,
    token_stride,
)

comptime LATENT_DIM = CACHE_DEPTH - ROPE_DEPTH  # 512: kv_lora_rank
comptime PAGE_SIZE = get_defined_int["page_size", 128]()
comptime OUT_DIM = 128  # one up-projection column block; see module docstring


def main() raises:
    with DeviceContext() as ctx:
        comptime if has_amd_gpu_accelerator():
            test_plan_gather_decompress(ctx, cache_length=500)
            # Larger page count: see the module docstring's Shapes section.
            test_plan_gather_decompress(ctx, cache_length=900)


def test_plan_gather_decompress(ctx: DeviceContext, cache_length: Int) raises:
    comptime batch_size = 1
    comptime MAX_CHUNKS = 1
    var seq_len = 9
    var num_keys = cache_length + seq_len  # not page-aligned

    print(
        "=== E2E plan+gather+decompress: cache_length=",
        cache_length,
        " seq_len=",
        seq_len,
        " num_keys=",
        num_keys,
        " ===",
    )
    seed(0)

    # ------------------------------------------------------------------
    # Step 1: paged latent+rope cache -- same contiguous single-sequence
    # layout every other paged-prefill test uses. Only the first
    # LATENT_DIM columns matter here; the rope tail is read directly by
    # the attention kernel elsewhere, not through this gather.
    # ------------------------------------------------------------------
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
    cache_lengths_host[0] = UInt32(cache_length)
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

    # ------------------------------------------------------------------
    # Step 2: stage 1 -- plan. Single sequence, single chunk (buffer_token_
    # size sized to the page-aligned total so nothing spills to a second
    # chunk).
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

    # Read the plan's own outputs back -- also a direct correctness check
    # on stage 1 itself, not just plumbing to feed stage 2.
    var buffer_row_offsets_host = alloc[UInt32](MAX_CHUNKS * (batch_size + 1))
    var cache_offsets_host = alloc[UInt32](MAX_CHUNKS * batch_size)
    var buffer_lengths_host = alloc[Int32](MAX_CHUNKS)
    ctx.enqueue_copy(buffer_row_offsets_host, buffer_row_offsets_device)
    ctx.enqueue_copy(cache_offsets_host, cache_offsets_device)
    ctx.enqueue_copy(buffer_lengths_host, buffer_lengths_device)
    ctx.synchronize()

    var expect_buffer_length = align_up(num_keys, PAGE_SIZE)
    print(
        "  plan: buffer_row_offsets=[",
        buffer_row_offsets_host[0],
        ",",
        buffer_row_offsets_host[1],
        "] cache_offsets=[",
        cache_offsets_host[0],
        "] buffer_lengths=[",
        buffer_lengths_host[0],
        "] (expect buffer_length=",
        expect_buffer_length,
        ")",
    )
    if Int(buffer_row_offsets_host[0]) != 0:
        raise Error(
            "plan: expected buffer_row_offsets[chunk=0,seq=0]=0, got "
            + String(buffer_row_offsets_host[0])
        )
    if Int(buffer_row_offsets_host[1]) != expect_buffer_length:
        raise Error(
            "plan: expected buffer_row_offsets[chunk=0,seq=1]="
            + String(expect_buffer_length)
            + ", got "
            + String(buffer_row_offsets_host[1])
        )
    if Int(cache_offsets_host[0]) != 0:
        raise Error(
            "plan: expected cache_offsets[chunk=0,seq=0]=0, got "
            + String(cache_offsets_host[0])
        )
    if Int(buffer_lengths_host[0]) != expect_buffer_length:
        raise Error(
            "plan: expected buffer_lengths[0]="
            + String(expect_buffer_length)
            + ", got "
            + String(buffer_lengths_host[0])
        )
    var buffer_length = Int(buffer_lengths_host[0])

    # ------------------------------------------------------------------
    # Step 3: stage 2 -- gather + decompress the first planned chunk.
    # ------------------------------------------------------------------
    var buffer_row_offsets_1d_tt = buffer_row_offsets_tt[0, :]
    var cache_offsets_1d_tt = cache_offsets_tt[0, :]

    var weight_host = alloc[Scalar[.bfloat16]](OUT_DIM * LATENT_DIM)
    randn[.bfloat16](weight_host, OUT_DIM * LATENT_DIM)
    var weight_device = ctx.enqueue_create_buffer[.bfloat16](
        OUT_DIM * LATENT_DIM
    )
    ctx.enqueue_copy(weight_device, weight_host)

    var k_latent_device = ctx.enqueue_create_buffer[.bfloat16](
        buffer_length * LATENT_DIM
    )
    var k_buffer_device = ctx.enqueue_create_buffer[.bfloat16](
        buffer_length * OUT_DIM
    )
    ctx.synchronize()

    var weight_tt = TileTensor(
        weight_device, row_major(Int64(OUT_DIM), Int64(LATENT_DIM))
    )
    var k_latent_tt = TileTensor(
        k_latent_device, row_major(Int64(buffer_length), Idx[LATENT_DIM])
    )
    var k_buffer_tt = TileTensor(
        k_buffer_device, row_major(Int64(buffer_length), Idx[OUT_DIM])
    )

    generic_flare_mla_decompress_k_cache_ragged_paged[
        target="gpu", dtype=DType.bfloat16
    ](
        buffer_row_offsets_1d_tt,
        cache_offsets_1d_tt,
        Int32(buffer_length),
        weight_tt,
        kv_collection,
        UInt32(0),  # layer_idx
        k_latent_tt,
        k_buffer_tt,
        ctx,
    )
    ctx.synchronize()

    var k_buffer_host = alloc[Scalar[.bfloat16]](buffer_length * OUT_DIM)
    ctx.enqueue_copy(k_buffer_host, k_buffer_device)
    ctx.synchronize()

    # ------------------------------------------------------------------
    # Step 4: reference. Extract each valid token's latent directly from
    # the host-side canonical (contiguous, page_base=0) cache content and
    # apply the same weight; rows past num_keys (the page-alignment tail)
    # are expected exactly zero, matching fill_paged_blocks_uniform's own
    # zero-fill of that region -- not assumed, asserted.
    # ------------------------------------------------------------------
    comptime atol: Float64 = 2e-2
    comptime rtol: Float64 = 2e-2
    var pstride = page_stride(PAGE_SIZE, CACHE_DEPTH)
    var tstride = token_stride(CACHE_DEPTH)
    var max_abs_err = Float64(0)
    var n_mismatch = 0

    for r in range(buffer_length):
        var expect_row = alloc[Float64](OUT_DIM)
        for o in range(OUT_DIM):
            expect_row[o] = 0.0

        if r < num_keys:
            var page_idx = r // PAGE_SIZE
            var tok_in_page = r % PAGE_SIZE
            var src_base = page_idx * pstride + tok_in_page * tstride
            for o in range(OUT_DIM):
                var acc = Float64(0)
                for d in range(LATENT_DIM):
                    var latent_val = blocks_host[src_base + d].cast[.float64]()
                    var w_val = weight_host[o * LATENT_DIM + d].cast[.float64]()
                    acc += latent_val * w_val
                expect_row[o] = acc

        for o in range(OUT_DIM):
            var actual = k_buffer_host[r * OUT_DIM + o].cast[.float64]()
            var expect = expect_row[o]
            var abs_err = abs(actual - expect)
            if abs_err > max_abs_err:
                max_abs_err = abs_err
            # Same combined threshold assert_almost_equal uses below --
            # a pure-atol check here previously flagged thousands of
            # ordinary bf16 rounding differences on large-magnitude sums
            # (e.g. actual=22.75 vs expect=22.7765, ~0.1% relative) as
            # "mismatches" while the real (rtol-aware) assert passed.
            if abs_err > atol + rtol * abs(expect):
                n_mismatch += 1
                if n_mismatch <= 8:
                    print(
                        "    mismatch r=",
                        r,
                        " o=",
                        o,
                        " actual=",
                        actual,
                        " expect=",
                        expect,
                    )
        expect_row.free()

    print(
        "  max_abs_err:",
        max_abs_err,
        " n_mismatch(>atol):",
        n_mismatch,
        " (",
        num_keys,
        " valid rows, ",
        buffer_length - num_keys,
        " zero-padding rows)",
    )

    for r in range(buffer_length):
        for o in range(OUT_DIM):
            var actual = k_buffer_host[r * OUT_DIM + o].cast[.float64]()
            var expect: Float64
            if r < num_keys:
                var page_idx = r // PAGE_SIZE
                var tok_in_page = r % PAGE_SIZE
                var src_base = page_idx * pstride + tok_in_page * tstride
                var acc = Float64(0)
                for d in range(LATENT_DIM):
                    var latent_val = blocks_host[src_base + d].cast[.float64]()
                    var w_val = weight_host[o * LATENT_DIM + d].cast[.float64]()
                    acc += latent_val * w_val
                expect = acc
            else:
                expect = 0.0
            assert_almost_equal(actual, expect, atol=atol, rtol=rtol)

    print("  RESULT: PASS")

    blocks_host.free()
    cache_lengths_host.free()
    lookup_table_host.free()
    input_row_offsets_host.free()
    buffer_row_offsets_host.free()
    cache_offsets_host.free()
    buffer_lengths_host.free()
    weight_host.free()
    k_buffer_host.free()

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
