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
inputs that drive the online-softmax rescale branches.
"""

from std.math import align_up, ceildiv
from std.random import randn, seed
from std.sys import get_defined_int
from std.utils.numerics import nan

from max.gpu.host import DeviceContext
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import (
    Idx,
    TileTensor,
    row_major,
)
from nn.attention.gpu.mha import mha_gpu_naive
from nn.attention.mha_mask import CausalMask, MHAMask, NullMask
from nn.attention.mha_utils import MHAConfig
from nn.attention.gpu.mla import AMDMLA, SM100MLA, flare_mla_decoding
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
)
from std.testing import assert_almost_equal

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


def main() raises:
    seed(1234)
    with DeviceContext() as ctx:
        comptime if ctx.target.is_amd_gpu() or ctx.target.is_nvidia_gpu():
            print("=" * 72)
            print("MLA decode fold at M = H*S > 64")
            print("=" * 72)
            print()

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

            print("=" * 72)
            print("ALL TESTS PASSED")
            print("=" * 72)
        else:
            print("Skipping: fold tests require an AMD or NVIDIA GPU")
