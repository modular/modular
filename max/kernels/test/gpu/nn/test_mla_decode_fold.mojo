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

"""Correctness tests for the MLA decode MTP fold at M = H*S > 64.

The MI355-enabled paged-variable target covers the fold only up to M=64
(H=16 S=4), and the B200-gated qkv_fp8 file's blocks never probe the K3
head count, so no enabled test ran the larger folds at the production
shapes. This file closes that gap for the shapes the K3 target-verify
decode runs at (nqh=12, nkvh=1, fp8 Q, fp8 paged KV, depth 576, bf16 out)
plus the fold-cap edge (H=16, S=5..8) and the H=8/S<=8 control. On MI355
the larger folds build two-tile launches (48/64-row M-tiles); on B200 the
same shapes exceed the SM100 fold cap (num_heads * q_max_seq_len <= 64)
and run its non-fold dispatch, one grid.y CTA per q position.

The oracle is the independent `mha_gpu_naive` reference (per batch, bf16
dequantized inputs, per-token causal positions) — never the kernel under
test. Coverage aimed at the failure modes of a row-tile split of the folded
query rows: uniform and mixed-width ragged batches (a short sequence's pad
rows must clamp on both the Q read and the output store), pad rows at
S=6/S=7 (M=72/84 live rows in the 96-slot two-tile launch), split-K with
empty partitions, NaN-poisoned output buffers, and signed / large-magnitude
inputs that drive the online-softmax rescale branches. The H=8 control also
runs at batch 4 with 32k caches, where the heuristic's wave target, not the
page cap, sets the split-K count.

Group 6 covers the short prompt chunk (8 < S <= 128) that Kimi K3 serves as
new user turns over a long cached context: the same fold, padded to a bucket
ceiling and spanning ceil(H*S/48) M-tiles, so a chunk reads the latent cache
once per tile instead of running MLA prefill's whole-context decompression.
Its host-side geometry (M-tile count, split, router ceiling) is pinned there
too, against the mirror in `mach/crates/engine-kv/src/dispatch.rs`.
"""

from std.math import align_up, ceildiv
from std.random import randn, seed
from std.sys import get_defined_int
from std.utils.numerics import nan

from max.gpu.host import DeviceContext, default_accelerator
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import (
    Idx,
    TileTensor,
    row_major,
)
from nn.attention.gpu.mha import mha_gpu_naive
from nn.attention.mha_mask import (
    CausalMask,
    MHAMask,
    NullMask,
    SlidingWindowCausalMask,
)
from nn.attention.mha_utils import MHAConfig
from nn.attention.gpu.mla import (
    AMDMLA,
    SM100MLA,
    flare_mla_decoding,
    mla_decode_max_seq_len,
    mla_mask_places_query_span,
)
from nn.attention.gpu.mha_decode_partition_heuristic import (
    amd_mla_decode_fold_num_m_tiles,
    hip_mha_decoding_num_partitions,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
)
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

# ===-----------------------------------------------------------------------===#
# Test parameters matching the K3 target-verify decode per rank
# ===-----------------------------------------------------------------------===#

comptime DEPTH = 576  # Q/K head dimension (512 latent + 64 rope)
comptime V_DEPTH = 512  # Output head dimension (depth - 64)
comptime PAGE_SIZE = get_defined_int["page_size", 128]()
comptime NUM_LAYERS = 1
comptime KV_NUM_HEADS = 1  # MLA has 1 KV head


def run_fold_case[
    mask_t: MHAMask,
    //,
    num_heads: Int,
](
    name: StringLiteral,
    cache_lengths: List[Int],
    seq_lens: List[Int],
    mask: mask_t,
    ctx: DeviceContext[],
    num_partitions: Optional[Int] = None,
    data_std: Float64 = 0.5,
    q_mean: Float64 = 0.0,
    k_tail_boost: Float64 = 0.0,
    atol: Float64 = 1e-1,
    rtol: Float64 = 2e-2,
) raises:
    """Native FP8 (Q=FP8, KV=FP8, output=BF16) MLA decode fold through the
    paged KV cache with per-batch query lengths.

    Uniform `seq_lens` exercise the uniform multi-Q fold; differing
    `seq_lens` exercise the runtime valid-rows path (the comptime
    `q_max_seq_len` is only the padding ceiling; the runtime per-sequence
    length drives the SRD row clamp, the per-token causal offset, the
    split-K stat writer, and the ragged reducer). The output buffer is
    NaN-poisoned before the launch so an unwritten live row cannot pass.

    The reference is `mha_gpu_naive` per batch with the DEQUANTIZED FP8 Q
    and K (the exact values the kernel consumes, so the comparison isolates
    the kernel's numerics from quantization error) and
    `num_keys = cache_len + this batch's seq_len`, so its
    `score_row = y + cache_len` matches the kernel's per-token
    `num_keys - seq_len + token`.
    """
    comptime q_type = DType.float8_e4m3fn
    comptime kv_type = DType.float8_e4m3fn
    comptime ref_type = DType.bfloat16

    var batch_size = len(cache_lengths)
    assert (
        len(seq_lens) == batch_size
    ), "cache_lengths and seq_lens must have same length"
    var q_max_seq_len = 0
    var total_q_tokens = 0
    for i in range(batch_size):
        if seq_lens[i] > q_max_seq_len:
            q_max_seq_len = seq_lens[i]
        total_q_tokens += seq_lens[i]

    print(
        t"test: {name}  batch_size: {batch_size}  num_heads: {num_heads} "
        t" q_max_seq_len: {q_max_seq_len}  total_q_tokens:"
        t" {total_q_tokens}  num_partitions:"
        t" {num_partitions.value() if num_partitions else -1}"
    )
    for i in range(batch_size):
        print(
            t"  batch {i} : cache_len= {cache_lengths[i]}  q_seq_len="
            t" {seq_lens[i]}"
        )

    var max_cache_len = 0
    var total_pages = 0
    for i in range(batch_size):
        if cache_lengths[i] > max_cache_len:
            max_cache_len = cache_lengths[i]
        total_pages += ceildiv(cache_lengths[i] + seq_lens[i], PAGE_SIZE)

    comptime scale = Float32(0.125)
    comptime group = num_heads

    # -----------------------------------------------------------------------
    # Step 1: Create the paged KV cache on host
    # -----------------------------------------------------------------------
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=DEPTH, is_mla=True
    )
    comptime kv_dim2 = 1

    var block_elems = (
        total_pages
        * kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    var blocks_host = List(length=block_elems, fill=Scalar[kv_type](0))
    var blocks_rand = List(length=block_elems, fill=Scalar[ref_type](0))
    randn(
        blocks_rand,
        mean=0.0,
        standard_deviation=data_std,
    )
    for i in range(block_elems):
        blocks_host[i] = blocks_rand[i].cast[kv_type]()
    _ = blocks_rand^

    var _page_stride = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )
    var _tok_stride = kv_params.num_heads * kv_params.head_size

    var cache_lengths_host = List(length=batch_size, fill=UInt32(0))
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    var max_num_keys_any_batch = 0
    for i in range(batch_size):
        var nk = cache_lengths[i] + seq_lens[i]
        if nk > max_num_keys_any_batch:
            max_num_keys_any_batch = nk
    # Pad to multiple of 8 for the LUT row-stride alignment (the
    # `PagedKVCache.populate` SIMD path uses chunks of up to 8 pages).
    var max_pages_per_batch = ceildiv(max_num_keys_any_batch, PAGE_SIZE)
    max_pages_per_batch = align_up(max_pages_per_batch, 8)
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = List(length=lut_size, fill=UInt32(0))

    var page_offset = 0
    for i in range(batch_size):
        var num_pages_i = ceildiv(cache_lengths[i] + seq_lens[i], PAGE_SIZE)
        for p in range(num_pages_i):
            lookup_table_host[i * max_pages_per_batch + p] = UInt32(
                page_offset + p
            )
        page_offset += num_pages_i

    # Zero out tail slots so a read past a batch's num_keys contributes
    # ~zero attention weight rather than garbage.
    var cur_page = 0
    for bi in range(batch_size):
        var num_keys_i = cache_lengths[bi] + seq_lens[bi]
        var num_pages_i = ceildiv(num_keys_i, PAGE_SIZE)
        for pg in range(num_pages_i):
            var valid_toks = num_keys_i - pg * PAGE_SIZE
            if valid_toks > PAGE_SIZE:
                valid_toks = PAGE_SIZE
            var base = cur_page * _page_stride + valid_toks * _tok_stride
            var zero_count = (PAGE_SIZE - valid_toks) * _tok_stride
            for z in range(zero_count):
                blocks_host[base + z] = 0
            cur_page += 1

    # Boost each batch's newest `seq_len` key rows so the causal tail
    # dominates every token's softmax. A mask-row bug that drops or shifts
    # the newest keys then changes the output by O(1) at any context length,
    # instead of by the tail's share of the softmax mass (4 of 32k keys is
    # invisible at any tolerance).
    if k_tail_boost > 0.0:
        var _page_base = 0
        for bi in range(batch_size):
            var num_keys_i = cache_lengths[bi] + seq_lens[bi]
            var num_pages_i = ceildiv(num_keys_i, PAGE_SIZE)
            for tok in range(cache_lengths[bi], num_keys_i):
                var page_idx, tok_in_page = divmod(tok, PAGE_SIZE)
                var base = (
                    _page_base + page_idx
                ) * _page_stride + tok_in_page * _tok_stride
                for d in range(_tok_stride):
                    # fp8 -> fp32 is exact, x8 is an exponent shift, and the
                    # fp8 round-trip of the shifted value is exact, so kernel
                    # and reference read identical boosted values.
                    blocks_host[base + d] = (
                        blocks_host[base + d].cast[.float32]()
                        * Float32(k_tail_boost)
                    ).cast[kv_type]()
            _page_base += num_pages_i

    # -----------------------------------------------------------------------
    # Step 2: Q tensor — generate BF16, keep copy, cast to FP8
    # -----------------------------------------------------------------------
    var q_size = total_q_tokens * num_heads * DEPTH
    var q_bf16_host = List(length=q_size, fill=Scalar[ref_type](0))
    randn(q_bf16_host, mean=q_mean, standard_deviation=data_std)
    var q_fp8_host = List(length=q_size, fill=Scalar[q_type](0))
    for i in range(q_size):
        q_fp8_host[i] = q_bf16_host[i].cast[q_type]()
    # The reference consumes the dequantized FP8 Q (what the kernel reads),
    # so the comparison isolates the kernel's numerics from Q quantization.
    var q_dequant_host = List(length=q_size, fill=Scalar[ref_type](0))
    for i in range(q_size):
        q_dequant_host[i] = q_fp8_host[i].cast[ref_type]()

    # -----------------------------------------------------------------------
    # Step 3: input_row_offsets — ragged per batch
    # -----------------------------------------------------------------------
    var row_offsets_host = List(length=batch_size + 1, fill=UInt32(0))
    row_offsets_host[0] = UInt32(0)
    for i in range(batch_size):
        row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(seq_lens[i])

    # -----------------------------------------------------------------------
    # Step 4: Output tensor (BF16), NaN-poisoned so an unwritten live row
    # cannot pass the comparison.
    # -----------------------------------------------------------------------
    var out_size = total_q_tokens * num_heads * V_DEPTH
    var out_host = List(length=out_size, fill=Scalar[ref_type](0))
    for i in range(out_size):
        out_host[i] = nan[ref_type]()

    # -----------------------------------------------------------------------
    # Step 5: Copy everything to device
    # -----------------------------------------------------------------------
    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)

    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    var q_device = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_device, q_fp8_host)

    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)

    var out_device = ctx.enqueue_create_buffer[ref_type](out_size)
    ctx.enqueue_copy(out_device, out_host)

    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Step 6: Build TileTensors and PagedKVCacheCollection on device
    # -----------------------------------------------------------------------
    var blocks_tt = TileTensor(
        blocks_device,
        row_major(
            Int64(total_pages),
            Idx[kv_dim2],
            Idx[NUM_LAYERS],
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
    var kv_collection = PagedKVCacheCollection[
        kv_type,
        kv_params,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ](
        blocks_tt.as_unsafe_any_origin(),
        cache_lengths_tt.as_imm().as_unsafe_any_origin(),
        lookup_table_tt.as_imm().as_unsafe_any_origin(),
        UInt32(q_max_seq_len),
        UInt32(max_cache_len),
    )

    var kv_cache = kv_collection.get_key_cache(0)

    var q_tt = TileTensor(
        q_device, row_major((total_q_tokens, Idx[num_heads], Idx[DEPTH]))
    )
    var out_tt = TileTensor(
        out_device, row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH]))
    )
    var row_offsets_tt = TileTensor(
        row_offsets_device, row_major(batch_size + 1)
    )

    # -----------------------------------------------------------------------
    # Step 7: Pre-compute scalar args and call the kernel
    # -----------------------------------------------------------------------
    print("  Launching native FP8 MLA decode fold kernel (ragged Q)...")

    # Each vendor's MLA backend is a required parameter. The AMD decode sizes
    # its own split-K grid and never reads the dispatch scalars, so it gets an
    # unread buffer (as `_mla_test_vendors.AMDVendor` does); SM100 reads them.
    comptime if ctx.target.is_amd_gpu():
        var unread_scalars = ctx.enqueue_create_buffer[DType.int64](3)
        flare_mla_decoding[
            kernels=AMDMLA,
            rank=3,
            config=MHAConfig[q_type](num_heads, DEPTH),
            ragged=True,
        ](
            out_tt,
            q_tt,
            kv_cache,
            mask,
            row_offsets_tt,
            scale,
            ctx,
            TileTensor(
                rebind[MutPointer[Int64, MutAnyOrigin]](
                    unread_scalars.unsafe_ptr()
                ),
                row_major((Idx[3],)),
            ),
            q_max_seq_len=q_max_seq_len,
            num_partitions=num_partitions,
        )
        _ = unread_scalars^
    else:
        var mla_args = MLADispatchScalarArgs[
            num_heads=num_heads,
            is_fp8_kv=(kv_type == .float8_e4m3fn),
        ](
            batch_size,
            max_cache_len,
            q_max_seq_len,
            ctx,
        )
        flare_mla_decoding[
            kernels=SM100MLA,
            rank=3,
            config=MHAConfig[q_type](num_heads, DEPTH),
            ragged=True,
        ](
            out_tt,
            q_tt,
            kv_cache,
            mask,
            row_offsets_tt,
            scale,
            ctx,
            mla_args.gpu_tile_tensor(),
            q_max_seq_len=q_max_seq_len,
            num_partitions=num_partitions,
        )
        _ = mla_args^

    ctx.synchronize()
    print("  Kernel completed successfully (no crash).")

    # -----------------------------------------------------------------------
    # Step 8: Numerical verification using mha_gpu_naive reference (BF16)
    # -----------------------------------------------------------------------
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    print("  Computing GPU naive reference per batch and comparing...")

    var total_checked = 0
    var max_abs_err = Float64(0)

    var q_token_offset = 0

    for b in range(batch_size):
        var cache_len = cache_lengths[b]
        var b_seq_len = seq_lens[b]
        var ref_num_keys = cache_len + b_seq_len

        # Extract contiguous BF16 K for this batch from the paged blocks.
        var k_b_size = ref_num_keys * KV_NUM_HEADS * DEPTH
        var k_b_host = List(length=k_b_size, fill=Scalar[ref_type](0))

        var page_base_b = 0
        for bi in range(b):
            page_base_b += ceildiv(cache_lengths[bi] + seq_lens[bi], PAGE_SIZE)

        for tok in range(ref_num_keys):
            var page_idx, tok_in_page = divmod(tok, PAGE_SIZE)
            var physical_page = page_base_b + page_idx

            var src_offset = (
                physical_page * _page_stride + tok_in_page * _tok_stride
            )
            var dst_offset = tok * KV_NUM_HEADS * DEPTH
            for d in range(KV_NUM_HEADS * DEPTH):
                # Dequantized FP8 K: the same values the kernel reads.
                k_b_host[dst_offset + d] = blocks_host[src_offset + d].cast[
                    ref_type
                ]()

        # BF16 Q for this batch: [1, b_seq_len, num_heads, depth]
        var q_b_size = b_seq_len * num_heads * DEPTH
        var q_b_host = List(length=q_b_size, fill=Scalar[ref_type](0))
        var q_batch_start = q_token_offset * num_heads * DEPTH
        for i in range(q_b_size):
            q_b_host[i] = q_dequant_host[q_batch_start + i]

        var ref_b_size = b_seq_len * num_heads * DEPTH
        var ref_b_host = List(length=ref_b_size, fill=Scalar[ref_type](0))

        var k_b_device = ctx.enqueue_create_buffer[ref_type](k_b_size)
        ctx.enqueue_copy(k_b_device, k_b_host)

        var q_b_device = ctx.enqueue_create_buffer[ref_type](q_b_size)
        ctx.enqueue_copy(q_b_device, q_b_host)

        var ref_b_device = ctx.enqueue_create_buffer[ref_type](ref_b_size)
        ctx.synchronize()

        var q_b_tt = TileTensor(
            q_b_device,
            row_major((Idx[1], b_seq_len, Idx[num_heads], Idx[DEPTH])),
        )
        var k_b_tt = TileTensor(
            k_b_device,
            row_major(
                (
                    Idx[1],
                    ref_num_keys,
                    Idx[KV_NUM_HEADS],
                    Idx[DEPTH],
                )
            ),
        )
        var ref_b_tt = TileTensor(
            ref_b_device,
            row_major((Idx[1], b_seq_len, Idx[num_heads], Idx[DEPTH])),
        )

        mha_gpu_naive(
            q_b_tt,
            k_b_tt,
            k_b_tt,
            mask,
            ref_b_tt,
            scale,
            1,  # batch_size
            b_seq_len,  # this batch's seq_len
            ref_num_keys,  # cache_len + this batch's seq_len
            num_heads,
            DEPTH,
            group,
            ctx,
        )

        ctx.synchronize()
        ctx.enqueue_copy(ref_b_host, ref_b_device)
        ctx.synchronize()

        # Compare the first V_DEPTH dims per head; the reference computes the
        # full 576 but the fold only outputs the latent 512 (no rope portion).
        for s in range(b_seq_len):
            var out_offset = (q_token_offset + s) * num_heads * V_DEPTH
            var ref_s_offset = s * num_heads * DEPTH
            for h in range(num_heads):
                for d in range(V_DEPTH):
                    var expect = ref_b_host[ref_s_offset + d + DEPTH * h].cast[
                        .float64
                    ]()
                    var actual = out_host[out_offset + V_DEPTH * h + d].cast[
                        .float64
                    ]()
                    var abs_err = abs(actual - expect)
                    if abs_err > max_abs_err:
                        max_abs_err = abs_err
                    if abs_err > atol / 2:
                        print(
                            t"batch {b} tok {s} head {h} dim {d} actual"
                            t" {actual} expect {expect}"
                        )
                    assert_almost_equal(actual, expect, atol=atol, rtol=rtol)

            total_checked += num_heads * V_DEPTH

        q_token_offset += b_seq_len

        _ = k_b_device
        _ = q_b_device
        _ = ref_b_device
        _ = ref_b_host^
        _ = q_b_host^
        _ = k_b_host^

    print(
        "  Verified:",
        total_checked,
        "elements, max_abs_err:",
        max_abs_err,
    )

    print("  PASS:", name, "\n")

    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = row_offsets_device
    _ = out_device
    _ = row_offsets_host^
    _ = lookup_table_host^
    _ = cache_lengths_host^
    _ = blocks_host^
    _ = out_host^
    _ = q_bf16_host^
    _ = q_fp8_host^
    _ = q_dequant_host^


def check_short_chunk_host_pins() raises:
    """Pins the short-chunk fold's host-side geometry (H=12, fp8 cache).

    The M-tile count and split are the launcher's and the device-graph capture
    key's; `mach/crates/engine-kv/src/dispatch.rs` mirrors both (see
    `hip_mla_short_chunk_fold_buckets_track_the_launch_ladder`), so these
    numbers must also hold there. 12 heads tile at 48 rows (4 tokens), and the
    launch pads the query rows to a bucket ceiling, so a raw width answers its
    bucket's count (S=23 is the 32-token bucket, 8 tiles).
    """
    comptime sm_count = 256
    for chunk, tiles in [
        (9, 4),
        (16, 4),
        (23, 8),
        (32, 8),
        (64, 16),
        (100, 32),
        (128, 32),
    ]:
        assert_equal(
            amd_mla_decode_fold_num_m_tiles(12, chunk),
            tiles,
            msg="H=12 short-chunk M-tile count",
        )
    for batch, chunk, tiles, want_np in [
        (4, 23, 8, 8),
        (4, 32, 8, 8),
        (1, 128, 32, 8),
        (4, 128, 32, 2),
    ]:
        assert_equal(
            hip_mha_decoding_num_partitions(
                batch,
                32768,
                12,
                sm_count,
                is_mla=True,
                num_m_tiles=tiles,
                q_seq_len=chunk,
            ),
            want_np,
            msg="H=12 short-chunk split",
        )
    assert_equal(
        hip_mha_decoding_num_partitions(
            4, 32768, 16, sm_count, is_mla=True, num_m_tiles=4, q_seq_len=16
        ),
        16,
        msg="H=16 chunk split",
    )
    # The router hands 8 < S <= 128 to decode only where a head-aligned tile
    # exists: 10 heads has none (64/48/32/16 are not multiples of 10), so it
    # keeps the prefill arm. SM100 folds up to 8 for every head count. A mask
    # that places the query span (everything but causal and null) keeps the
    # exact-width MTP fold, whatever the head count.
    comptime if default_accelerator().is_amd_gpu():
        assert_equal(
            mla_decode_max_seq_len[DType.float8_e4m3fn, 12, False](),
            128,
            msg="H=12 router ceiling",
        )
        assert_equal(
            mla_decode_max_seq_len[DType.float8_e4m3fn, 16, False](),
            128,
            msg="H=16 router ceiling",
        )
        assert_equal(
            mla_decode_max_seq_len[DType.float8_e4m3fn, 10, False](),
            8,
            msg="H=10 keeps the MTP fold",
        )
        assert_equal(
            mla_decode_max_seq_len[DType.float8_e4m3fn, 32, False](),
            1,
            msg="H=32 is S=1 only",
        )
        assert_equal(
            mla_decode_max_seq_len[DType.bfloat16, 12, False](),
            1,
            msg="bf16 is S=1 only",
        )
        assert_equal(
            mla_decode_max_seq_len[DType.float8_e4m3fn, 12, False, True](),
            8,
            msg="span-placing mask caps at the MTP fold",
        )
        assert_false(mla_mask_places_query_span["causal"](), msg="causal")
        assert_false(mla_mask_places_query_span["null"](), msg="null")
        assert_true(
            mla_mask_places_query_span["sliding_window_causal"](),
            msg="sliding window",
        )
    else:
        assert_equal(
            mla_decode_max_seq_len[DType.float8_e4m3fn, 12, False](),
            8,
            msg="SM100 fan-out ceiling",
        )


def run_short_chunk_group(ctx: DeviceContext[]) raises:
    """Runs Group 6, the AMD-only padded short-chunk fold and its gates.

    SM100 folds up to 8 query rows for every head count, so these widths and
    the host gates they probe exist only on the AMD decode path.
    """
    # -----------------------------------------------------------
    # Group 6: H=12 short prompt chunks over a cached context. 8 < S
    # <= 128 pads the query rows up to a bucket ceiling (16/32/64/128)
    # and spans ceil(H*S/48) 48-row M-tiles, so a chunk reads the
    # latent cache once per M-tile instead of running MLA prefill's
    # per-chunk decompression. Every case runs the production causal
    # mask; the mixed-width batches keep each sequence's live rows
    # (H*seq_len) separate from the padded tile rows, so a tile that is
    # only partly alive, and a fully dead one, both appear.
    # -----------------------------------------------------------
    print("--- Group 6: H=12 short prompt chunks (S in 9..128) ---")

    run_fold_case[12](
        "h12_chunk_s9_b1_causal_nk32768",
        [32768],
        [9],
        CausalMask(),
        ctx,
    )
    run_fold_case[12](
        "h12_chunk_s16_b1_causal_nk32768",
        [32768],
        [16],
        CausalMask(),
        ctx,
    )
    run_fold_case[12](
        "h12_chunk_s23_b1_causal_nk32768",
        [32768],
        [23],
        CausalMask(),
        ctx,
    )
    run_fold_case[12](
        "h12_chunk_s32_b1_causal_nk32768",
        [32768],
        [32],
        CausalMask(),
        ctx,
    )
    run_fold_case[12](
        "h12_chunk_s64_b1_causal_nk32768",
        [32768],
        [64],
        CausalMask(),
        ctx,
    )
    run_fold_case[12](
        "h12_chunk_s100_b1_causal_nk32768",
        [32768],
        [100],
        CausalMask(),
        ctx,
    )
    run_fold_case[12](
        "h12_chunk_s128_b1_causal_nk32768",
        [32768],
        [128],
        CausalMask(),
        ctx,
    )
    # Mixed widths: the padded ceiling is the longest row, so the
    # shorter sequence drops its pad rows at a different tile.
    run_fold_case[12](
        "h12_chunk_mixed_b2_s23",
        [32768, 32768],
        [23, 20],
        CausalMask(),
        ctx,
    )
    run_fold_case[12](
        "h12_chunk_mixed_b2_s128",
        [32768, 32768],
        [128, 125],
        CausalMask(),
        ctx,
    )
    # 9 live tokens of 16 slots: tiles 1..3 straddle / follow the live
    # row count (H*9 = 108 rows over 48-row tiles).
    run_fold_case[12](
        "h12_chunk_mixed_b2_s9",
        [32768, 32768],
        [9, 6],
        CausalMask(),
        ctx,
    )
    # Mid-size caches (heuristic np, the reducer's ragged remap over
    # several M-tiles) and deep ones (np across many partitions).
    run_fold_case[12](
        "h12_chunk_s16_b1_causal_nk4096",
        [4096],
        [16],
        CausalMask(),
        ctx,
    )
    run_fold_case[12](
        "h12_chunk_s16_b1_causal_nk131072",
        [131072],
        [16],
        CausalMask(),
        ctx,
    )
    run_fold_case[12](
        "h12_chunk_s128_b1_causal_nk4096",
        [4096],
        [128],
        CausalMask(),
        ctx,
    )
    run_fold_case[12](
        "h12_chunk_s128_b1_causal_nk131072",
        [131072],
        [128],
        CausalMask(),
        ctx,
    )
    run_fold_case[12](
        "h12_chunk_mixed_b2_deep",
        [131072, 131072],
        [128, 5],
        CausalMask(),
        ctx,
    )
    # num_keys <= 256 returns np=1, so these run the multi-tile fold
    # through the direct ragged store: 4 M-tiles at H=12, every tile's
    # base row and its pad rows inside one 16-token bucket.
    run_fold_case[12](
        "h12_chunk_s16_b2_np1",
        [30, 128],
        [16, 16],
        CausalMask(),
        ctx,
    )
    # A batch of 3 with a different cache length and width per row.
    run_fold_case[12](
        "h12_chunk_b3_np1_ragged",
        [0, 40, 96],
        [9, 16, 32],
        CausalMask(),
        ctx,
    )
    # 16 heads tile at 64 rows (4 tokens); 8 heads tile at 64 rows
    # (8 tokens), the 64-row geometry the drafter class uses. Both
    # ragged, so the pad rows of a short row sit inside a live tile of
    # the longer one.
    run_fold_case[16](
        "h16_chunk_mixed_b2",
        [4096, 512],
        [16, 9],
        CausalMask(),
        ctx,
    )
    run_fold_case[8](
        "h8_chunk_mixed_b2",
        [8192, 256],
        [16, 12],
        CausalMask(),
        ctx,
    )
    # Ragged cache lengths at chunk widths: the heuristic sees only the
    # longest cache, while each request scans its own num_keys.
    run_fold_case[12](
        "h12_chunk_b4_ragged_cache",
        [32768, 512, 4096, 131072],
        [9, 16, 100, 128],
        CausalMask(),
        ctx,
    )
    # A short row inside a wide bucket: seq_len=16 of a 128-token
    # ceiling leaves 28 of its 32 M-tiles with no live rows at all.
    run_fold_case[12](
        "h12_chunk_mixed_b2_deadtiles",
        [2048, 40],
        [128, 16],
        CausalMask(),
        ctx,
    )
    # Explicit split counts: 2 (the one-wave target at this shape) and
    # 64 (well past it), both over 8 M-tiles of 4 tokens.
    run_fold_case[12](
        "h12_chunk_s32_np2",
        [32768],
        [32],
        CausalMask(),
        ctx,
        num_partitions=Optional(2),
    )
    run_fold_case[12](
        "h12_chunk_s32_np64",
        [32768],
        [32],
        CausalMask(),
        ctx,
        num_partitions=Optional(64),
    )

    # --- Ragged-Q gates the host `raise` still rejects ---
    # Dense Q only reaches the exact-width MTP fold, so the remaining
    # gates of the padded-chunk fold need ragged Q: a head count with
    # no head-aligned M-tile, a mask that places the query span, and a
    # width past the bucket ladder. These are the probes the qkv_fp8
    # negative block also carries; that file is B200-only, so pin them
    # here where MI355 runs them.
    with assert_raises():
        # 10 heads has no head-aligned tile under the 64-row cliff
        # (16/32/48/64 are not multiples of 10).
        run_fold_case[10](
            "h10_chunk_s16_must_raise",
            [4096],
            [16],
            CausalMask(),
            ctx,
        )
    with assert_raises():
        # A sliding-window causal mask reads its key span off the
        # query width, which a padded bucket would overstate.
        run_fold_case[12](
            "h12_chunk_s16_sw_must_raise",
            [4096],
            [16],
            SlidingWindowCausalMask[256](),
            ctx,
        )
    with assert_raises():
        # S=129 is past the 128-token bucket ceiling.
        run_fold_case[12](
            "h12_chunk_s129_must_raise",
            [4096],
            [129],
            CausalMask(),
            ctx,
        )


def main() raises:
    seed(1234)
    with DeviceContext() as ctx:
        comptime if ctx.target.is_amd_gpu() or ctx.target.is_nvidia_gpu():
            print("=" * 72)
            print("MLA decode fold at M = H*S > 64")
            print("=" * 72)
            print()

            check_short_chunk_host_pins()

            # -----------------------------------------------------------
            # Group 1: H=12 uniform S in {6,7,8} — the K3 target-verify
            # shapes (M=72/84/96, two 48-row M-tiles each). S=6 and S=7
            # carry pad rows (72/84 live rows of 96 slots); S=8 is exact.
            # CAUSAL is the discriminating mask (per-token score_row);
            # NO_MASK pairs with it for the unmasked reduction. Contexts
            # span the split-K boundary (np>1) and include batches whose
            # short caches leave partitions empty.
            # -----------------------------------------------------------
            print("--- Group 1: H=12 uniform S in {6,7,8} ---")

            run_fold_case[12](
                "h12_s8_b1_causal_nk256",
                [256],
                [8],
                CausalMask(),
                ctx,
            )
            run_fold_case[12](
                "h12_s8_b1_causal_nk4096",
                [4096],
                [8],
                CausalMask(),
                ctx,
            )
            # Causal-tail-dominant: the newest 8 keys carry the softmax
            # mass, so a mask-row bug that masks/shifts the newest keys
            # fails here by O(1), at a production-shaped context.
            run_fold_case[12](
                "h12_s8_b1_causal_tailboost",
                [4096],
                [8],
                CausalMask(),
                ctx,
                k_tail_boost=8.0,
            )
            run_fold_case[12](
                "h12_s8_b1_null_nk4096",
                [4096],
                [8],
                NullMask(),
                ctx,
            )
            run_fold_case[12](
                "h12_s8_b3_causal_nk2048",
                [2048, 2048, 2048],
                [8, 8, 8],
                CausalMask(),
                ctx,
            )
            # Asymmetric contexts: the 30-key batch leaves most of the 32
            # heuristic partitions empty.
            run_fold_case[12](
                "h12_s8_b4_causal_mixed_cl",
                [30, 2048, 8192, 32768],
                [8, 8, 8, 8],
                CausalMask(),
                ctx,
            )
            # The production point: b4/32k (heuristic np=32).
            run_fold_case[12](
                "h12_s8_b4_causal_nk32768",
                [32768, 32768, 32768, 32768],
                [8, 8, 8, 8],
                CausalMask(),
                ctx,
            )
            # At batch 8 the heuristic steps down to the np=16 rung (one
            # wave of 8 batches x 2 tiles): the only batch>4 fold shape.
            run_fold_case[12](
                "h12_s8_b8_causal_nk4096",
                [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096],
                [8, 8, 8, 8, 8, 8, 8, 8],
                CausalMask(),
                ctx,
            )
            # 128-partition reducer (np=128) with the fold.
            run_fold_case[12](
                "h12_s8_b1_causal_nk65536",
                [65536],
                [8],
                CausalMask(),
                ctx,
            )
            # Explicit partition count with empty partitions: 38 keys over
            # 16 partitions leaves most of them key-less.
            run_fold_case[12](
                "h12_s8_b1_causal_nk30_np16",
                [30],
                [8],
                CausalMask(),
                ctx,
                num_partitions=Optional(16),
            )

            run_fold_case[12](
                "h12_s6_b1_causal_nk256",
                [256],
                [6],
                CausalMask(),
                ctx,
            )
            run_fold_case[12](
                "h12_s6_b1_causal_nk4096",
                [4096],
                [6],
                CausalMask(),
                ctx,
            )
            run_fold_case[12](
                "h12_s6_b1_null_nk256",
                [256],
                [6],
                NullMask(),
                ctx,
            )
            run_fold_case[12](
                "h12_s7_b1_causal_nk256",
                [256],
                [7],
                CausalMask(),
                ctx,
            )
            run_fold_case[12](
                "h12_s7_b1_causal_nk4096",
                [4096],
                [7],
                CausalMask(),
                ctx,
            )

            # -----------------------------------------------------------
            # Group 2: H=12 mixed-width ragged batches — the runtime
            # valid-rows path (per-request seq_len < q_max_seq_len), the
            # layout K3's mixed-width verify uses. Asymmetric widths and
            # contexts so a row-tile offset bug shows: a short sequence's
            # pad rows must clamp on the Q read and the output store, and
            # a fully-dead second tile (seq_len=1 at S=8) must write
            # nothing outside its live rows.
            # -----------------------------------------------------------
            print("--- Group 2: H=12 mixed-width ragged batches ---")

            run_fold_case[12](
                "h12_mixed_s8_wide_ctx",
                [256, 1024, 4096, 32768],
                [6, 7, 8, 8],
                CausalMask(),
                ctx,
            )
            run_fold_case[12](
                "h12_mixed_s8_deep_ctx",
                [65536, 32768, 8192, 1024],
                [8, 6, 7, 8],
                CausalMask(),
                ctx,
            )
            # Tiny caches (np=1, direct ragged store): batch 0 has 12 live
            # rows of 96 slots — the second row tile is fully dead.
            run_fold_case[12](
                "h12_mixed_s8_tiny_np1",
                [30, 64, 96, 120],
                [1, 6, 8, 3],
                CausalMask(),
                ctx,
            )
            # The dead-tile case above runs at np=1; with split-K the dead
            # tile also writes no per-partition stats or partials, and the
            # reducer must skip its rows (it early-returns on pad rows).
            # Batch 1's seq_len=1 row (12 live rows) leaves tile 1 fully
            # dead at heuristic np=64.
            run_fold_case[12](
                "h12_mixed_s8_deadtile_splitk",
                [32768, 30],
                [8, 1],
                CausalMask(),
                ctx,
            )
            run_fold_case[12](
                "h12_mixed_s8_null_deep",
                [256, 65536, 1024, 32768],
                [2, 8, 5, 8],
                NullMask(),
                ctx,
            )
            run_fold_case[12](
                "h12_mixed_s6",
                [2048, 30, 640, 4096],
                [6, 3, 1, 6],
                CausalMask(),
                ctx,
            )

            # -----------------------------------------------------------
            # Group 3: H=16, S in {5..8} — the fold-cap edge (M=80..128).
            # These split across two 48/64-row M-tiles; they must stay
            # correct under any future tile-geometry change.
            # -----------------------------------------------------------
            print("--- Group 3: H=16 fold-cap edge (S in {5..8}) ---")

            run_fold_case[16](
                "h16_s5_b1_causal_nk4096",
                [4096],
                [5],
                CausalMask(),
                ctx,
            )
            run_fold_case[16](
                "h16_s6_b1_causal_nk4096",
                [4096],
                [6],
                CausalMask(),
                ctx,
            )
            run_fold_case[16](
                "h16_s7_b1_causal_nk4096",
                [4096],
                [7],
                CausalMask(),
                ctx,
            )
            run_fold_case[16](
                "h16_s8_b1_causal_nk4096",
                [4096],
                [8],
                CausalMask(),
                ctx,
            )
            run_fold_case[16](
                "h16_s8_b2_causal_nk2048",
                [2048, 2048],
                [8, 8],
                CausalMask(),
                ctx,
            )
            run_fold_case[16](
                "h16_mixed_s8",
                [1024, 30, 2048, 8192],
                [8, 5, 6, 8],
                CausalMask(),
                ctx,
            )
            # An H=16 seq_len<=3 row (48 live rows) leaves the 64-row tile 1
            # fully dead; the S in {5..8} cases above never produce one.
            run_fold_case[16](
                "h16_mixed_s8_deadtile",
                [4096, 30],
                [8, 3],
                CausalMask(),
                ctx,
            )

            # -----------------------------------------------------------
            # Group 4: H=8, S in {7,8} — the BM=64 control. M=56/64 <= 64
            # must stay on the single-tile ladder (the gate that protects
            # the drafter-class shape), so these must keep passing under
            # any change to the M>64 split gate.
            # -----------------------------------------------------------
            print("--- Group 4: H=8 BM=64 control (S in {7,8}) ---")

            run_fold_case[8](
                "h8_s7_b1_causal_nk4096",
                [4096],
                [7],
                CausalMask(),
                ctx,
            )
            run_fold_case[8](
                "h8_s8_b1_causal_nk4096",
                [4096],
                [8],
                CausalMask(),
                ctx,
            )

            # The two cases above run at batch 1, where the split-K count is
            # page-capped (4096 keys = 16 split pages), so a change to the
            # partition heuristic is invisible to them. The DSpark drafter runs
            # this H=8 fold at batch 4 with S=7/8, where the one-wave target
            # binds instead (np = 64 at 32768 keys, 128 split pages).
            run_fold_case[8](
                "h8_s7_b4_causal_nk32768",
                [32768, 32768, 32768, 32768],
                [7, 7, 7, 7],
                CausalMask(),
                ctx,
            )
            run_fold_case[8](
                "h8_s8_b4_causal_nk32768",
                [32768, 32768, 32768, 32768],
                [8, 8, 8, 8],
                CausalMask(),
                ctx,
            )
            # Ragged cache lengths at batch 4: the heuristic sees only the
            # longest cache, while each request scans its own `num_keys`, so
            # partitions are unevenly filled. Lengths are not multiples of the
            # 128-row key tile, so per-partition key ranges land off-tile.
            run_fold_case[8](
                "h8_s7_b4_ragged_cache_causal_nk32768",
                [32768, 20000, 30001, 12345],
                [7, 7, 7, 7],
                CausalMask(),
                ctx,
            )
            # At 32k random keys every output element sits far below the
            # tolerance, so the cases above cannot see a reduce that drops or
            # misweights a partition. With the causal tail boosted, the last
            # partition carries the softmax mass and such an error moves the
            # output by O(1).
            run_fold_case[8](
                "h8_s7_b4_causal_nk32768_tailboost",
                [32768, 32768, 32768, 32768],
                [7, 7, 7, 7],
                CausalMask(),
                ctx,
                k_tail_boost=8.0,
            )
            run_fold_case[8](
                "h8_s7_b4_ragged_cache_causal_nk32768_tailboost",
                [32768, 20000, 30001, 12345],
                [7, 7, 7, 7],
                CausalMask(),
                ctx,
                k_tail_boost=8.0,
            )

            # -----------------------------------------------------------
            # Group 4b: H=12 cells with a DMA producer warp on MI355 (32-row
            # at S=2, 48-row at S=3/4 and the two-tile S=8). Forced split
            # counts give partitions of 3, 4 and 5 key tiles, so the
            # producer runs its two-tile loop with an odd and an even tail;
            # ragged cache lengths leave each partition's last tile partial.
            # -----------------------------------------------------------
            print("--- Group 4b: H=12 producer-warp tile counts ---")

            # np=4: 1500/2000/2500/3100 keys -> 3/4/5/7 tiles per partition.
            run_fold_case[12](
                "h12_s8_b4_ragged_np4_tiles3to7",
                [1492, 1992, 2492, 3092],
                [8, 8, 8, 8],
                CausalMask(),
                ctx,
                num_partitions=Optional(4),
            )
            run_fold_case[12](
                "h12_s4_b4_ragged_np4_tiles3to7",
                [1496, 1996, 2496, 3096],
                [4, 4, 4, 4],
                CausalMask(),
                ctx,
                num_partitions=Optional(4),
            )
            run_fold_case[12](
                "h12_s3_b4_ragged_np4_tiles3to7",
                [1497, 1997, 2497, 3097],
                [3, 3, 3, 3],
                CausalMask(),
                ctx,
                num_partitions=Optional(4),
            )
            run_fold_case[12](
                "h12_s2_b4_ragged_np4_tiles3to7",
                [1498, 1998, 2498, 3098],
                [2, 2, 2, 2],
                CausalMask(),
                ctx,
                num_partitions=Optional(4),
            )
            # np=64 over long caches: 3/4/5/2 tiles per partition.
            run_fold_case[12](
                "h12_s8_b4_ragged_np64_long",
                [24000, 28000, 39000, 12000],
                [8, 8, 8, 8],
                CausalMask(),
                ctx,
                num_partitions=Optional(64),
            )
            run_fold_case[12](
                "h12_s2_b4_ragged_np64_long",
                [24000, 28000, 39000, 12000],
                [2, 2, 2, 2],
                CausalMask(),
                ctx,
                num_partitions=Optional(64),
            )
            # np=1: one partition per request, hundreds of tiles.
            run_fold_case[12](
                "h12_s8_b4_ragged_np1_long",
                [20000, 30001, 12345, 32760],
                [8, 8, 8, 8],
                CausalMask(),
                ctx,
                num_partitions=Optional(1),
            )
            run_fold_case[12](
                "h12_s2_b4_ragged_np1_long",
                [20000, 30001, 12345, 32760],
                [2, 2, 2, 2],
                CausalMask(),
                ctx,
                num_partitions=Optional(1),
            )

            # -----------------------------------------------------------
            # Group 5: edge data. Near-empty caches (num_keys == S and
            # S+1) exercise the shortest causal tails; large-magnitude
            # inputs drive the online-softmax max/rescale branches; a
            # negative-mean Q keeps the score mass signed.
            # -----------------------------------------------------------
            print("--- Group 5: edge data ---")

            run_fold_case[12](
                "h12_s8_b1_causal_nk0",
                [0],
                [8],
                CausalMask(),
                ctx,
            )
            run_fold_case[12](
                "h12_s8_b1_causal_nk1",
                [1],
                [8],
                CausalMask(),
                ctx,
            )
            run_fold_case[12](
                "h12_s8_b1_causal_bigstd",
                [4096],
                [8],
                CausalMask(),
                ctx,
                data_std=2.0,
                atol=5e-1,
                rtol=8e-2,
            )
            run_fold_case[12](
                "h12_s8_b1_causal_negmean",
                [4096],
                [8],
                CausalMask(),
                ctx,
                q_mean=-1.0,
            )

            comptime if ctx.target.is_amd_gpu():
                run_short_chunk_group(ctx)

            print("=" * 72)
            print("ALL TESTS PASSED")
            print("=" * 72)
        else:
            print("Skipping: fold tests require an AMD or NVIDIA GPU")
